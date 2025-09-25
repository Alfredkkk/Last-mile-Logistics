# drl_comodal.py
# Deep RL for joint ride-hailing & package delivery (no zoning)
# Environment: diamond region (L1 metric), homogeneous Poisson arrivals, Poisson package field
# Author: ChatGPT (PyTorch PPO)

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# Global knobs (easy tuning)
# =========================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# City / motion
R = 10.0                 # half "L1 radius" of diamond region; feasible points satisfy |x|+|y| <= R
V = 1.0                  # speed (distance units per minute)
DT = 0.5                 # minutes per step
HORIZON_MIN = 240.0      # episode length in minutes (e.g., 4 hours)
STEPS_PER_EP = int(HORIZON_MIN / DT)

# Demand & revenue
LAMBDA = 0.20            # ride arrival rate per minute (Poisson, spatially uniform over region)
R_PICK_ALPHA = 0.25      # pickup visibility radius parameter r = alpha * R / sqrt(2) (L1-constraint approx)
RT = 8.0                 # per-distance revenue for rides
RP = 1.0                 # revenue per delivered package
GAMMA_PACK = 0.8         # package spatial intensity (packages per unit area); area = R^2 for diamond in L1 model
RIDE_TTL_MINUTES = 3    # ride request time to live in minutes

# DRL obs/action shaping
MAX_VISIBLE_RIDES = 5    # keep top-K closest pickups
K_NEAREST_PACK = 10      # encode nearest K packages
DISCOUNT = 0.99
PPO_STEPS = 4096
PPO_MINI_BATCH = 256
PPO_EPOCHS = 4
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
LR = 3e-4
UPDATES = 200

EVAL_EPISODES = 5        # evaluation batch after each update

rng = np.random.default_rng(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


# ==============
# Util functions
# ==============
def manhattan(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(a[0]-b[0]) + abs(a[1]-b[1]))

def l1_inside(x: float, y: float, r: float) -> bool:
    return abs(x) + abs(y) <= r

def sample_uniform_point_in_diamond(R: float) -> np.ndarray:
    # Rejection sampling in square [-R, R]^2 with L1 constraint
    while True:
        x = rng.uniform(-R, R)
        y = rng.uniform(-R, R)
        if abs(x) + abs(y) <= R:
            return np.array([x, y], dtype=np.float32)

def project_to_diamond(p: np.ndarray, R: float) -> np.ndarray:
    # If |x|+|y|>R, project to boundary along direction to origin
    s = abs(p[0]) + abs(p[1])
    if s <= R:
        return p
    if s == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    return p * (R / s)

def step_towards(from_pt: np.ndarray, to_pt: np.ndarray, max_dist: float) -> Tuple[np.ndarray, float, bool]:
    """Move along Manhattan shortest path: first x, then y (or vice versa); return (new_pos, traveled, reached)"""
    x0, y0 = from_pt
    x1, y1 = to_pt
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    d = dx + dy
    if d <= max_dist:
        return to_pt.copy(), d, True
    # Move along x first
    move_x = min(dx, max_dist)
    sign_x = np.sign(x1 - x0)
    x_new = x0 + sign_x * move_x
    remaining = max_dist - move_x
    if remaining > 1e-8:
        move_y = min(dy, remaining)
        sign_y = np.sign(y1 - y0)
        y_new = y0 + sign_y * move_y
    else:
        y_new = y0
    p = np.array([x_new, y_new], dtype=np.float32)
    return p, max_dist, False


# ========================
# Environment (no zoning)
# ========================
@dataclass
class RideReq:
    pickup: np.ndarray
    dropoff: np.ndarray
    trip_len: float

class CoModalEnv:
    """
    - Region: diamond |x|+|y| <= R (L1 metric)
    - Packages: spatial Poisson with intensity gamma; N ~ Poisson(gamma * R^2), positions IID uniform in diamond
    - Ride arrivals: Poisson(LAMBDA * DT) per step, each with (origin, destination) uniform in diamond
    - Vehicle:
        * cannot deliver packages while carrying passenger (pickup->dropoff)
        * revenue: RP per delivered package; RT * (distance with passenger)
    - Action (discrete): 0 = continue delivering package; i=1..MAX_VISIBLE_RIDES = accept ith visible ride (if available)
      When busy enroute to pickup/dropoff, action is ignored (auto-continue).
    - Observation: vector with normalized features:
        * pos (x/R, y/R), time_frac, flags (to_pickup, with_pass)
        * K nearest packages: for each, (dx/R, dy/R, l1/R)
        * up to K rides visible: for each, (dx_pick/R, dy_pick/R, l1_pick/R, dx_drop/R, dy_drop/R, l1_trip/R)
        * counts: remaining_pkg / (1 + E[N]), current_visible / MAX_VISIBLE_RIDES
      Invalid slots are zero-padded. We also return an action mask for invalid ride indices.
    """
    def __init__(self,
                 R: float = R,
                 v: float = V,
                 dt: float = DT,
                 lam: float = LAMBDA,
                 gamma_pack: float = GAMMA_PACK,
                 rp: float = RP,
                 rt: float = RT,
                 r_pick_alpha: float = R_PICK_ALPHA,
                 ride_ttl_minutes: float=RIDE_TTL_MINUTES): #ride request time to live in minutes
        self.R = R
        self.v = v
        self.dt = dt
        self.lam = lam
        self.gamma = gamma_pack
        self.rp = rp
        self.rt = rt
        # max pickup radius (L1) following r = alpha * R / sqrt(2); we keep L1 constraint
        self.r_pick = r_pick_alpha * R / math.sqrt(2.0)

        self.max_visible = MAX_VISIBLE_RIDES
        self.k_pack = K_NEAREST_PACK
        self.ride_ttl_steps = max(1, int(round(ride_ttl_minutes / self.dt)))
        
        self.reset()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        # sample packages
        area = self.R ** 2  # for L1 diamond in the paper
        n_pkg = rng.poisson(self.gamma * area)
        n_pkg = max(1, int(n_pkg))  # ensure non-empty
        self.packages = [sample_uniform_point_in_diamond(self.R) for _ in range(n_pkg)]
        self.packages = np.array(self.packages, dtype=np.float32)

        self.pos = np.array([0.0, 0.0], dtype=np.float32)
        self.t = 0.0
        self.ride_buffer: List[Tuple[RideReq, int]] = []
        self.to_pickup: Optional[np.ndarray] = None
        self.with_passenger: bool = False
        self.drop_target: Optional[np.ndarray] = None

        self.revenue_cum = 0.0
        self._steps = 0
        obs, mask = self._get_obs()
        return obs, mask

    def _append_new_ride(self, reqs: List[RideReq]):
        """Bring the new ride request to this buffer."""
        if not reqs:
            return
        ttl0 = self.ride_ttl_steps
        self.ride_buffer.extend([(r, ttl0) for r in reqs])

    def _sample_rides_this_step(self) -> List[RideReq]:
        k = rng.poisson(self.lam * self.dt)
        reqs = []
        for _ in range(int(k)):
            pk = sample_uniform_point_in_diamond(self.R)
            dp = sample_uniform_point_in_diamond(self.R)
            reqs.append(RideReq(pk, dp, manhattan(pk, dp)))
        return reqs

    def _visible_rides(self) -> List[RideReq]:
        # only show rides with pickup within r_pick (L1) and TTL>0
        cand = [r for (r, ttl) in self.ride_buffer
                if ttl > 0 and manhattan(self.pos, r.pickup) <= self.r_pick]
        cand.sort(key=lambda r: manhattan(self.pos, r.pickup))
        return cand[:self.max_visible]

    def _nearest_package(self) -> Optional[np.ndarray]:
        if len(self.packages) == 0:
            return None
        dists = np.abs(self.packages - self.pos).sum(axis=1)
        idx = int(np.argmin(dists))
        return self.packages[idx].copy()


    def _deliver_if_arrived(self):
        if len(self.packages) == 0:
            return 0
        # Deliver packages exactly at current location (robust with small threshold)
        dists = np.abs(self.packages - self.pos).sum(axis=1)
        hit = np.where(dists < 1e-6)[0]
        delivered = len(hit)
        if delivered > 0:
            # deliver all that are exactly here (batch)
            self.revenue_cum += self.rp * delivered
            self.packages = np.delete(self.packages, hit, axis=0)
        return delivered

    def _get_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        # Visible rides & nearest packages
        visible = self._visible_rides()
        # Packages features
        pack_feats = []
        if len(self.packages) > 0:
            d = np.abs(self.packages - self.pos).sum(axis=1)
            order = np.argsort(d)[:self.k_pack]
            for j in order:
                rel = (self.packages[j] - self.pos) / self.R
                pack_feats.extend([rel[0], rel[1], d[j] / self.R])
        # pad
        while len(pack_feats) < 3 * self.k_pack:
            pack_feats.append(0.0)

        ride_feats = []
        for r in visible:
            relp = (r.pickup - self.pos) / self.R
            reld = (r.dropoff - self.pos) / self.R
            ride_feats.extend([relp[0], relp[1], manhattan(self.pos, r.pickup) / self.R,
                               reld[0], reld[1], r.trip_len / self.R])
        while len(ride_feats) < 6 * self.max_visible:
            ride_feats.append(0.0)

        time_frac = self.t / HORIZON_MIN
        flags = [1.0 if self.to_pickup is not None else 0.0,
                 1.0 if self.with_passenger else 0.0]

        counts = [len(self.packages) / (1.0 + self.gamma * (self.R ** 2)),
                  len(visible) / float(self.max_visible)]

        core = [self.pos[0] / self.R, self.pos[1] / self.R, time_frac] + flags + counts
        obs = np.array(core + pack_feats + ride_feats, dtype=np.float32)

        # Action mask: 0 always valid; i>0 valid if i<=len(visible)
        mask = np.zeros(1 + self.max_visible, dtype=np.float32)
        mask[0] = 1.0
        for i in range(len(visible)):
            mask[1 + i] = 1.0
        return obs, mask

    def step(self, action: int):
        reward = 0.0
        done = False

        # add fresh arrivals
        self._append_new_ride(self._sample_rides_this_step())

        # If busy towards pickup or with passenger, ignore action (auto-continue)
        if self.to_pickup is not None:
            # move towards pickup
            new_pos, d, reached = step_towards(self.pos, self.to_pickup, self.v * self.dt)
            self.pos = project_to_diamond(new_pos, self.R)
            if reached:
                self.to_pickup = None
                self.with_passenger = True
            # cannot deliver while enroute to pickup
        elif self.with_passenger:
            # move towards dropoff; accrue ride revenue per distance traveled
            new_pos, d, reached = step_towards(self.pos, self.drop_target, self.v * self.dt)
            self.pos = project_to_diamond(new_pos, self.R)
            reward += self.rt * d    # ride distance revenue in this step
            self.revenue_cum += self.rt * d
            if reached:
                self.with_passenger = False
                self.drop_target = None
        else:
            # free: can choose to deliver or accept a visible ride
            visible = self._visible_rides()
            if action > 0 and action <= len(visible):
                chosen = visible[action - 1]
                # remove chosen from buffer
                # (remove by identity)
                for i, (r, ttl) in enumerate(self.ride_buffer):
                    if r is chosen:
                        self.ride_buffer.pop(i)
                        break
                # set pickup/drop targets
                self.to_pickup = chosen.pickup.copy()
                self.drop_target = chosen.dropoff.copy()
                # move towards pickup immediately this step
                new_pos, d, reached = step_towards(self.pos, self.to_pickup, self.v * self.dt)
                self.pos = project_to_diamond(new_pos, self.R)
                if reached:
                    self.to_pickup = None
                    self.with_passenger = True
            else:
                # deliver: go to nearest package (if any)
                target = self._nearest_package()
                if target is not None:
                    new_pos, d, reached = step_towards(self.pos, target, self.v * self.dt)
                    self.pos = project_to_diamond(new_pos, self.R)
                    if reached:
                        delivered = self._deliver_if_arrived()
                        reward += self.rp * delivered
                # else idle at current location

        # small clean-up: remove stale rides outside pickup radius (we keep buffer but they vanish after 1 step) and TTL Decay
        # simple model: unaccepted rides expire by end of step
        if self.ride_buffer:
            new_buf = []
            for (r, ttl) in self.ride_buffer:
                if ttl > 0:
                    new_buf.append((r, ttl))
            self.ride_buffer = new_buf

        self.t += self.dt
        self._steps += 1

        # Deliver any packages exactly at position (numerical safety)
        if self.to_pickup is None and not self.with_passenger:
            delivered = self._deliver_if_arrived()
            reward += self.rp * delivered

        if self.t >= HORIZON_MIN:
            done = True
        if (len(self.packages) == 0) and (not self.with_passenger) and (self.to_pickup is None):
            # all packages done and not carrying a passenger -> end early
            done = True

        obs, mask = self._get_obs()
        return obs, reward, done, {}, mask

    @property
    def obs_dim(self):
        # core (3 + 2 flags + 2 counts) + 3*K + 6*MAX_VISIBLE
        return (3 + 2 + 2) + 3 * self.k_pack + 6 * self.max_visible

    @property
    def act_dim(self):
        return 1 + self.max_visible


# =================
# Baseline policies
# =================
def run_episode(env: CoModalEnv, policy=None, greedy=False) -> float:
    obs, mask = env.reset()
    total = 0.0
    for _ in range(STEPS_PER_EP):
        if policy is None:
            # default: deliver-only baseline
            action = 0
        else:
            with torch.no_grad():
                o = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                m = torch.tensor(mask, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                logits, _ = policy(o)
                # mask invalid
                invalid = (m < 0.5)
                logits = logits.masked_fill(invalid, -1e9)
                if greedy:
                    action = int(torch.argmax(logits, dim=-1).item())
                else:
                    probs = torch.softmax(logits, dim=-1)
                    action = int(torch.multinomial(probs, 1).item())
        obs, r, done, _, mask = env.step(action)
        total += r
        if done:
            break
    return total


def baseline_nearby_rule(env: CoModalEnv, pickup_alpha=R_PICK_ALPHA, drop_bias=0.5) -> float:
    """
    Heuristic: accept a ride if:
      - pickup within r_pick
      - and dropoff is closer (L1) than 'drop_bias * nearest remaining package distance'
    Else deliver.
    """
    obs, mask = env.reset()
    total = 0.0
    for _ in range(STEPS_PER_EP):
        # decode visible rides
        # simple access via env methods
        visible = env._visible_rides()
        action = 0
        if len(visible) > 0:
            # nearest package distance
            nn = env._nearest_package()
            nn_d = manhattan(env.pos, nn) if nn is not None else 0.0
            chosen_idx = -1
            best_score = -1e9
            for i, r in enumerate(visible):
                pk_d = manhattan(env.pos, r.pickup)
                dp_d = manhattan(env.pos, r.dropoff)
                # favor close pickup and dropoff closer than package
                score = -pk_d + (nn_d - dp_d) * drop_bias
                if score > best_score:
                    best_score = score
                    chosen_idx = i
            if best_score > 0:
                action = 1 + chosen_idx
        obs, r, done, _, mask = env.step(action)
        total += r
        if done:
            break
    return total


# ==============
# PPO Components
# ==============
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        hid = 256
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
        )
        self.pi = nn.Linear(hid, act_dim)
        self.v  = nn.Linear(hid, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.pi(h), self.v(h)


def ppo_update(policy: ActorCritic, optimizer, batch, clip_eps=CLIP_EPS):
    obs, act, old_logp, ret, adv, mask = batch
    logits, v = policy(obs)
    # mask invalid actions
    invalid = (mask < 0.5)
    logits = logits.masked_fill(invalid, -1e9)
    dist = torch.distributions.Categorical(logits=logits)
    logp = dist.log_prob(act)
    ratio = torch.exp(logp - old_logp)

    clip_adv = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    pi_loss = -(torch.min(ratio * adv, clip_adv)).mean()
    v_loss = ((ret - v.squeeze(-1)) ** 2).mean()

    ent = dist.entropy().mean()
    loss = pi_loss + VF_COEF * v_loss - ENT_COEF * ent

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return pi_loss.item(), v_loss.item(), ent.item(), loss.item()


def collect_rollout(env: CoModalEnv, policy: ActorCritic, steps: int):
    obs_buf, act_buf, rew_buf, val_buf, logp_buf, mask_buf, done_buf = [], [], [], [], [], [], []
    obs, mask = env.reset()
    for _ in range(steps):
        o = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        m = torch.tensor(mask, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits, v = policy(o)
            invalid = (m < 0.5)
            logits = logits.masked_fill(invalid, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a).squeeze(0)
            val = v.squeeze(0).squeeze(-1)
        obs2, r, done, _, mask2 = env.step(int(a.item()))

        obs_buf.append(o.squeeze(0).cpu().numpy())
        mask_buf.append(m.squeeze(0).cpu().numpy())
        act_buf.append(a.cpu().numpy())
        rew_buf.append(r)
        val_buf.append(val.cpu().numpy())
        logp_buf.append(logp.cpu().numpy())
        done_buf.append(done)

        obs, mask = obs2, mask2
        if done:
            obs, mask = env.reset()
    # compute GAE-lambda = 1 (pure MC) for simplicity
    rets, advs = [], []
    ret = 0.0
    for r, v, d in zip(reversed(rew_buf), reversed(val_buf), reversed(done_buf)):
        if d:
            ret = 0.0
        ret = r + DISCOUNT * ret
        rets.append(ret)
        advs.append(ret - v)
    rets.reverse(); advs.reverse()
    # to tensors
    obs_t  = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=DEVICE)
    act_t  = torch.tensor(np.array(act_buf).squeeze(-1), dtype=torch.long, device=DEVICE)
    ret_t  = torch.tensor(np.array(rets), dtype=torch.float32, device=DEVICE)
    adv_t  = torch.tensor(np.array(advs), dtype=torch.float32, device=DEVICE)
    logp_t = torch.tensor(np.array(logp_buf), dtype=torch.float32, device=DEVICE)
    mask_t = torch.tensor(np.array(mask_buf), dtype=torch.float32, device=DEVICE)

    # normalize advantages
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    return obs_t, act_t, logp_t, ret_t, adv_t, mask_t


def make_minibatches(batch, batch_size):
    N = batch[0].shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    for i in range(0, N, batch_size):
        j = idx[i:i+batch_size]
        yield tuple(x[j] for x in batch)


def evaluate_all(env: CoModalEnv, policy: ActorCritic):
    # DRL (greedy)
    drl = np.mean([run_episode(env, policy, greedy=True) for _ in range(EVAL_EPISODES)])
    # Heuristics
    pure = np.mean([run_episode(env, policy=None) for _ in range(EVAL_EPISODES)])
    heur = np.mean([baseline_nearby_rule(env) for _ in range(EVAL_EPISODES)])
    return drl, heur, pure


def main():
    env = CoModalEnv()
    policy = ActorCritic(env.obs_dim, env.act_dim).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    print(f"Device: {DEVICE}, obs_dim={env.obs_dim}, act_dim={env.act_dim}")
    print("Start training PPO...")
    for upd in range(1, UPDATES + 1):
        batch = collect_rollout(env, policy, PPO_STEPS)
        for _ in range(PPO_EPOCHS):
            for mb in make_minibatches(batch, PPO_MINI_BATCH):
                pi_l, v_l, ent, tot = ppo_update(policy, optimizer, mb, CLIP_EPS)
        if upd % 5 == 0:
            drl, heur, pure = evaluate_all(env, policy)
            print(f"[Upd {upd:03d}] DRL={drl:8.2f} | heur={heur:8.2f} | pure={pure:8.2f} "
                  f"| pi={pi_l:.3f} v={v_l:.3f} ent={ent:.3f}")

    # Final evaluation
    drl, heur, pure = evaluate_all(env, policy)
    print("="*80)
    print(f"Final evaluation over {EVAL_EPISODES} eps (4h each):")
    print(f"  DRL (greedy):        {drl:.2f}")
    print(f"  Nearby heuristic:    {heur:.2f}")
    print(f"  Pure delivery only:  {pure:.2f}")

if __name__ == "__main__":
    main()
