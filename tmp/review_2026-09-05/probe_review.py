"""Read notebook definitions and probe behavior without installing training dependencies.
No notebook source or historical result is modified.
"""
import ast
import contextlib
import copy
import csv
import dataclasses
import itertools
import json
import math
import pathlib
import sys
import types
import typing
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
OUT = pathlib.Path(__file__).parent
results = []

def record(name, **kwargs):
    row = dict(test=name, **kwargs)
    results.append(row)
    print(json.dumps(row, ensure_ascii=False, default=str))

def load_defs(filename):
    nb = json.loads(pathlib.Path(filename).read_text(encoding='utf-8'))
    code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    tree = ast.parse(code)
    ns = dict(np=np, pd=pd, math=math, dataclass=dataclasses.dataclass,
              **{n:getattr(typing,n) for n in ['List','Tuple','Optional','Dict']})
    # Defaults come from the actual notebook's parameter cell.
    for node in ast.parse(''.join(nb['cells'][1]['source'])).body:
        if isinstance(node, ast.Assign):
            try: exec(compile(ast.Module(body=[node],type_ignores=[]), filename,'exec'),ns)
            except NameError: pass
    ns['HOURLY_MULTIPLIER'] = np.ones(24)
    names = ['manhattan','l1_inside','project_to_diamond','step_towards','RideReq','CoModalEnv',
             'baseline_nearby_rule','baseline_nearby_rule_voronoi','baseline_four_zone',
             'l1_zone_id','_l1_distance','_build_distance_matrix_L1','_make_eval_env_from']
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
            exec(compile(ast.Module(body=[node],type_ignores=[]),filename,'exec'),ns)
    return ns, tree

def set_packages(env, points):
    env.pkg_pos_all = np.array(points,dtype=np.float32)
    env.pkg_remaining_idx = list(range(len(points)))
    env.pkg_delivered = np.zeros(len(points),dtype=bool)
    env.packages = env.pkg_pos_all.copy()

for filename in ['experiment.ipynb','NonStationary/experiment2.ipynb']:
    ns, tree = load_defs(filename)
    Env, Ride = ns['CoModalEnv'], ns['RideReq']
    def make():
        e=Env(seed=123,lam=0,r_pick_alpha=0.2)
        set_packages(e,[[4,0]])
        return e

    # Real environment: identical observations hide accepted passenger destination.
    a,b=make(),make()
    a.with_passenger=b.with_passenger=True
    a.drop_target=np.array([.05,0],dtype=np.float32)
    b.drop_target=np.array([1,0],dtype=np.float32)
    equal=np.array_equal(a._get_obs()[0],b._get_obs()[0])
    ra=a.step(0)[1];rb=b.step(0)[1]
    record('hidden_passenger_target',file=filename,observations_identical=equal,
           next_rewards=[float(ra),float(rb)],passenger_flags=[a.with_passenger,b.with_passenger])

    # Real environment: last passenger step counted as delivery time.
    record('time_phase',file=filename,passenger_step_minutes=a.dt,
           recorded_ride_minutes=a.time_rides_min,recorded_delivery_minutes=a.time_delivery_min)

    # Actual TTL screening: reference ETA does not enforce vehicle arrival deadline.
    e=make();pk=np.array([4,0],dtype=np.float32);dp=np.array([0,1],dtype=np.float32)
    e.ride_buffer=[(Ride(pk,dp,5.),1)]
    visible=len(e._visible_rides());eta=ns['manhattan'](e.pos,pk)/e.v
    e.step(1)
    while e.to_pickup is not None and e.t<30:e.step(0)
    record('ttl_vehicle_feasibility',file=filename,visible=visible,remaining_ttl=.5,
           vehicle_eta=eta,pickup_time=e.t,accepted=e.accepted_rides)

    # Actual distance function used by FOUR_ZONE.
    record('four_zone_distance_rounding',file=filename,
           distance_049=ns['_l1_distance']((0.,0.),(.49,0.)),
           distance_051=ns['_l1_distance']((0.,0.),(.51,0.)))

    # Script only substitutes the route solver, not the policy or environment.
    # Each zone has one package, so route order is uniquely determined.
    ns['solve_zone_tsp_L1']=lambda points,start:list(range(len(points)))
    class ScriptedEnv(Env):
        def reset(self,seed=None):
            super().reset(seed=seed)
            set_packages(self,[[.05,0],[-.1,0],[0,4]])
            self.events=[]
            self.offered_first=False
            self.offered_second=False
            self.chain=False
            return self._get_obs()
        def _sample_rides_this_step(self):
            if self.pkg_delivered[0] and not self.offered_first:
                self.offered_first=True
                pk=np.array([.05,0],dtype=np.float32)
                dp=np.array([-.01,.2],dtype=np.float32)
                return [Ride(pk,dp,ns['manhattan'](pk,dp))]
            if (self.chain and self.accepted_rides==1 and not self.with_passenger
                and self.to_pickup is None and not self.offered_second):
                self.offered_second=True
                pk=self.pos.copy();dp=np.array([-.02,0],dtype=np.float32)
                return [Ride(pk,dp,ns['manhattan'](pk,dp))]
            return []
        def step(self,action):
            old_acc=self.accepted_rides;old_pass=self.with_passenger
            target=self._nearest_package()
            before=self.pos.copy()
            out=super().step(action)
            if self.accepted_rides>old_acc:self.events.append(['accept',float(self.t),self.drop_target.tolist()])
            if old_pass and not self.with_passenger:self.events.append(['drop',float(self.t),self.pos.tolist()])
            if out[3]['delivered_step']:self.events.append(['deliver',float(self.t),self.pos.tolist()])
            if not old_pass and old_acc and target is not None and action==0 and self.to_pickup is None:
                self.events.append(['target_after_ride',float(self.t),target.tolist()])
            return out
    e=ScriptedEnv(seed=1,lam=0)
    ns['baseline_four_zone'](e,seed=1)
    record('four_zone_destination',file=filename,events=e.events[:8])
    # Force second demand only for this fixture; real baseline still runs unchanged.
    old_reset=e.reset
    def reset_chain(seed=None):
        ret=old_reset(seed);e.chain=True;return ret
    e.reset=reset_chain
    ns['baseline_four_zone'](e,seed=1)
    record('four_zone_chaining',file=filename,events=e.events[:9])

    # Static check of actual rollout time coverage.
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='collect_rollout')
    resets=[n.lineno for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and isinstance(n.func.value,ast.Name) and n.func.value.id=='env' and n.func.attr=='reset']
    record('rollout_coverage',file=filename,reset_lines=resets,
           maximum_sampled_episode_min=ns['PPO_STEPS']*ns['DT'],evaluation_horizon_min=ns['HORIZON_MIN'])

    # Check regression fixes directly with environment actions and seeds.
    a=Env(seed=33,lam=2);b=Env(seed=33,lam=2)
    seed_equal=np.array_equal(a.packages,b.packages)
    for _ in range(10):
        oa=a.step(0);ob=b.step(0)
        seed_equal=seed_equal and np.array_equal(oa[0],ob[0]) and oa[1]==ob[1]
    record('seed_reproducibility',file=filename,passed=bool(seed_equal))

    e=make();e.pos=np.array([4,0],dtype=np.float32)
    old=Ride(np.array([4.1,0],np.float32),np.array([0,1],np.float32),5.1)
    new=Ride(np.array([4.01,0],np.float32),np.array([0,2],np.float32),6.01)
    e.ride_buffer=[(old,10)];e._sample_rides_this_step=lambda:[new]
    e.step(1)
    record('action_index_regression',file=filename,accepted_observed_ride=bool(np.array_equal(e.drop_target,old.dropoff)))

    a=Env(seed=71,lam=40,r_pick_alpha=.25);b=Env(seed=71,lam=40,r_pick_alpha=.4)
    equal=True
    for _ in range(200):
        action=1 if a._visible_rides() else 0
        oa=a.step(action);ob=b.step(action)
        equal=equal and np.array_equal(oa[0],ob[0]) and np.array_equal(oa[4],ob[4]) and oa[1]==ob[1]
    record('alpha_saturation',file=filename,alpha_values=[.25,.4],identical_200_steps=bool(equal))

    if filename.startswith('NonStationary/'):
        e=Env(seed=71,hourly_multiplier=np.linspace(.5,1.5,24))
        evaluation=ns['_make_eval_env_from'](e)
        record('custom_hourly_profile_clone',file=filename,
               profile_preserved=bool(np.array_equal(e.hourly_multiplier,evaluation.hourly_multiplier)),
               train_first_hour=float(e.hourly_multiplier[0]),eval_first_hour=float(evaluation.hourly_multiplier[0]))

    # Test trainer control flow with lightweight collaborators; no actual PPO training.
    trainer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_policy_brief')
    class DummyPolicy:
        def __init__(self,*args):self.updates=0
        def to(self,*args):return self
        def train(self):pass
        def parameters(self):return []
    def update_stub(policy,*args):
        policy.updates+=1
        return (0.,0.,0.,0.)
    def eval_stub(env,policy,**kwargs):
        metric=dict(avg_reward=policy.updates,avg_t=1.,avg_rate=policy.updates,avg_ep_rate=policy.updates,
                    finish_rate=1.,avg_finish_time=1.)
        return {name:metric.copy() for name in ['drl','heur','heur_vor','four_zone','pure','pure_or']}
    ns.update(ActorCritic=DummyPolicy,DEVICE='cpu',optim=types.SimpleNamespace(Adam=lambda *a,**kw:object()),
              collect_rollout=lambda *args:(None,[],0.,0.),make_minibatches=lambda *args:[None],
              ppo_update=update_stub,evaluate_all=eval_stub,PPO_EPOCHS=1)
    exec(compile(ast.Module(body=[trainer],type_ignores=[]),filename,'exec'),ns)
    with contextlib.redirect_stdout(__import__('io').StringIO()):
        policy,metrics=ns['train_policy_brief'](make(),updates=7,eval_every=5)
    record('stale_final_evaluation',file=filename,model_updates=policy.updates,
           returned_metrics_update=metrics['drl']['avg_reward'])

# Exact existing analysis functions on a counterexample, without plotting.
for filename in ['analysis.ipynb','NonStationary/analysis2.ipynb']:
    nb=json.loads(pathlib.Path(filename).read_text(encoding='utf-8'))
    code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    tree=ast.parse(code);ns=dict(np=np,pd=pd)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='best_alpha_series')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),filename,'exec'),ns)
    frame=pd.DataFrame({'GAMMA_PACK':[.5,.5],'algo':['PURE_OR','PURE_OR'],
        'LAMBDA':[10,10],'R_PICK_ALPHA':[.1,.2],'rate':[1.,3.]})
    _,rates=ns['best_alpha_series'](frame,.5,'PURE_OR')
    mapping={g:float(np.array([30,50])[np.argmin(abs(np.array([30,50])-g*60.5))])
             for g in [.33,.50,.67,.83,1.]}
    record('analysis_grouping',file=filename,pure_or_selected=float(rates[0]),
           pure_or_expected_average=2.,scatter_gamma_to_n=mapping)

record('pickup_area_fraction',alpha=.2,actual_interior_fraction=(.2*5.5/math.sqrt(2))**2/5.5**2,
       paper_fraction=.2**2,ttl_alpha_saturation=.19*5*math.sqrt(2)/5.5)

# Closed vs open route objective counterexample with exact enumeration.
pts=[(0.,0.),(1.,0.),(-1.,0.),(0.,2.)]
def d(a,b):return abs(a[0]-b[0])+abs(a[1]-b[1])
perms=list(itertools.permutations(range(1,len(pts))))
costs=[(p,sum(d(pts[a],pts[b]) for a,b in zip((0,)+p,p)),d(pts[p[-1]],pts[0])) for p in perms]
record('closed_vs_open_objective',all_routes=[{'route':p,'open':c,'closed':c+last} for p,c,last in costs])
(OUT/'probe_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
