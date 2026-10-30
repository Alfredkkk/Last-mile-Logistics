
# =================
# Baseline policies
# =================
def run_episode(env: CoModalEnv, policy=None, greedy=False, seed: Optional[int] = None) -> Tuple[float, float, float, str, int, float, float]:
    obs, mask = env.reset(seed=seed)
    total = 0.0
    terminal_time = None
    ended_reason = "unknown"
    acc = 0
    t_rides = 0.0
    t_delivery = 0.0
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
        obs, r, done, info, mask = env.step(action)
        total += r
        if done:
            terminal_time = float(info.get("terminal_time_min", env.t))
            ended_reason = str(info.get("ended_reason", env._ended_reason or "unknown"))
            acc = int(info.get("accepted_rides", getattr(env, "accepted_rides", 0)))
            t_rides = float(info.get("time_rides_min", 0.0))
            t_delivery = float(info.get("time_delivery_min", 0.0))
            break
    if terminal_time is None:
        terminal_time = float(env.t)
        t_rides = float(getattr(env, "time_rides_min", 0.0))
        t_delivery = float(getattr(env, "time_delivery_min", 0.0))
        
    revenue_rate = total / terminal_time if terminal_time > 0 else 0.0
    
    return total, terminal_time, revenue_rate, ended_reason, acc, t_rides, t_delivery


def baseline_nearby_rule(env: CoModalEnv,
                         pickup_alpha=R_PICK_ALPHA,
                         drop_bias=0.5,
                         switch_grace_steps: Optional[int] = None,
                         seed: Optional[int] = None
                         ) -> Tuple[float, float, float, str, int, float, float]:
    # pickup_alpha and drop_bias are legacy parameters; this baseline uses env.r_pick and delivery-reference visibility directly.
    obs, mask = env.reset(seed=seed)
    total = 0.0
    terminal_time = None
    ended_reason = "unknown"
    acc = 0
    t_rides = 0.0
    t_delivery = 0.0

    grace_left = 0

    for _ in range(STEPS_PER_EP):
        visible = env._visible_rides()
        action = 0  # Default: go to the nearest package
        
        gs = SWITCH_GRACE_STEPS if switch_grace_steps is None else int(switch_grace_steps)
        
        #Pre-grace
        # If the nearest package can be delivered in the next step, open the window in advance
        nn = env._nearest_package()
        if nn is not None:
            dist_to_nn = manhattan(env.pos, nn)
            if grace_left == 0 and dist_to_nn <= (env.v * env.dt):
                grace_left = max(1, gs)
                

        if grace_left > 0:
            # Inside the window: as long as there is a qualified order, accept the one nearest to the visibility reference.
            if len(visible) > 0:
                vis_ref = env._visibility_reference_point()
                # if cap is set, only consider rides with pickup distance <= cap from the visibility reference
                if HEUR_SOFT_PICK_CAP is not None:
                    cand = [(i, r) for i, r in enumerate(visible)
                            if manhattan(vis_ref, r.pickup) <= HEUR_SOFT_PICK_CAP]
                else:
                    cand = list(enumerate(visible))

                if cand:
                    # pick the candidate with the smallest pickup distance from the visibility reference
                    best_i = min(cand, key=lambda ir: manhattan(vis_ref, ir[1].pickup))[0]
                    action = 1 + best_i
                    grace_left = 0
                else:
                    # no ride within the cap → keep window open and continue delivering
                    grace_left -= 1
            else:
                grace_left -= 1

        # Execute action
        obs, r, done, info, mask = env.step(action)
        total += r

        delivered_cnt = int(info.get("delivered_step", 0))
        if delivered_cnt > 0:
            # After delivering the package: open a new window (including the current and subsequent SWITCH_GRACE_STEPS steps)
            grace_left = max(1, gs)


        if done:
            terminal_time = float(info.get("terminal_time_min", env.t))
            ended_reason  = str(info.get("ended_reason", env._ended_reason or "unknown"))
            acc = int(info.get("accepted_rides", getattr(env, "accepted_rides", 0)))
            t_rides = float(info.get("time_rides_min", 0.0))
            t_delivery = float(info.get("time_delivery_min", 0.0))
            break

    if terminal_time is None:
        terminal_time = float(env.t)
    revenue_rate = (total / terminal_time) if terminal_time > 0 else 0.0
    return total, terminal_time, revenue_rate, ended_reason, acc, t_rides, t_delivery


def baseline_nearby_rule_voronoi(env: CoModalEnv,
                                 pickup_alpha=R_PICK_ALPHA,
                                 drop_bias=0.5,
                                 switch_grace_steps: Optional[int] = None,
                                 seed: Optional[int] = None
                                 ) -> Tuple[float, float, float, str, int, float, float]:
    """
    Switching policy (N=n) + L1-Voronoi dropoff filter:
      - Only consider accepting rides within a window of a few steps after delivering a package.
      - Filter candidate passengers first: the dropoff must not fall into the L1-Voronoi cell of a delivered package.
      - If there are still candidates after filtering, accept the pickup nearest to the delivery-reference point; otherwise, continue delivering the nearest package.
    Returns: (total_reward, terminal_time_min, revenue_rate, ended_reason, accepted_rides)
    """
    obs, mask = env.reset(seed=seed)
    total = 0.0
    terminal_time = None
    ended_reason = "unknown"
    acc = 0
    grace_left = 0
    t_rides = 0.0
    t_delivery = 0.0

    gs = SWITCH_GRACE_STEPS if switch_grace_steps is None else int(switch_grace_steps)

    for _ in range(STEPS_PER_EP):
        visible = env._visible_rides()
        action = 0

        # If the nearest package can be delivered in the next step, open the window in advance
        nn = env._nearest_package()
        if nn is not None:
            dist_to_nn = manhattan(env.pos, nn)
            if grace_left == 0 and dist_to_nn <= (env.v * env.dt):
                grace_left = max(1, gs)

        if grace_left > 0:
            # First, use the Voronoi rule to filter out rides whose dropoff falls into a delivered package's cell
            cand = [(i, r) for i, r in enumerate(visible) if env.dropoff_allowed(r.dropoff)]
            if len(cand) > 0:
                # Accept the nearest pickup from the visibility reference while preserving the original visible index for the action.
                vis_ref = env._visibility_reference_point()
                best_i = min(cand, key=lambda ir: manhattan(vis_ref, ir[1].pickup))[0]
                action = 1 + best_i
                grace_left = 0
            else:
                grace_left -= 1

        obs, r, done, info, mask = env.step(action)
        total += r

        delivered_cnt = int(info.get("delivered_step", 0))   # Note: your env uses "delivered_step"
        if delivered_cnt > 0:
            grace_left = max(1, gs)

        if done:
            terminal_time = float(info.get("terminal_time_min", env.t))
            ended_reason  = str(info.get("ended_reason", env._ended_reason or "unknown"))
            acc = int(info.get("accepted_rides", getattr(env, "accepted_rides", 0)))
            t_rides = float(info.get("time_rides_min", 0.0))
            t_delivery = float(info.get("time_delivery_min", 0.0))
            break

    if terminal_time is None:
        terminal_time = float(env.t)
        t_rides = float(getattr(env, "time_rides_min", 0.0))
        t_delivery = float(getattr(env, "time_delivery_min", 0.0))
        
    revenue_rate = (total / terminal_time) if terminal_time > 0 else 0.0
    return total, terminal_time, revenue_rate, ended_reason, acc, t_rides, t_delivery


def baseline_pure_ortools(env: CoModalEnv, seed: Optional[int] = None) -> Tuple[float, float, float, str, int, float, float]:
    """
    Pure delivery baseline: Use OR-Tools to solve an open L1 route for all package points,
    then force the environment to strictly follow this order to clear packages one by one (completely ignoring passengers).
    Returns: (total_reward, terminal_time_min, revenue_rate, ended_reason, accepted_rides=0, time_rides, time_delivery)
    """
    obs, mask = env.reset(seed=seed)
    total = 0.0
    terminal_time = None
    ended_reason = "unknown"
    t_rides = 0.0
    t_delivery = 0.0

    # ---- Collect package points & starting point ----
    pkgs = env.packages                     # view: updates with remaining set, but we only use the initial full set to plan the route once
    start_xy = (float(env.pos[0]), float(env.pos[1]))
    pkg_pts = [(float(p[0]), float(p[1])) for p in pkgs]

    # ---- Build integer distance matrix (with scaling) ----
    SCALE = 1000  # Avoid short edges being rounded to 0
    def _l1_int(a, b):
        return int(round((abs(a[0]-b[0]) + abs(a[1]-b[1])) * SCALE))

    def _build_dist(pts):
        n = len(pts)
        mat = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    mat[i][j] = _l1_int(pts[i], pts[j])
        return mat

    # ---- Open route: keep departure costs, set the artificial return arc to zero ----
    def _solve_tsp(order_pts):
        if not order_pts:
            return []
        all_pts = [start_xy] + order_pts
        dist = _build_dist(all_pts)
        n_nodes = len(all_pts)
        manager = pywrapcp.RoutingIndexManager(n_nodes, 1, 0)     # 1 vehicle, depot=0
        routing = pywrapcp.RoutingModel(manager)

        def cb(from_idx, to_idx):
            i = manager.IndexToNode(from_idx)
            j = manager.IndexToNode(to_idx)
            return 0 if routing.IsEnd(to_idx) else dist[i][j]

        cb_idx = routing.RegisterTransitCallback(cb)
        routing.SetArcCostEvaluatorOfAllVehicles(cb_idx)

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.FromSeconds(2)  # Sufficient for small instances

        sol = routing.SolveWithParameters(params)
        if sol is None:
            # Fallback: Sort by L1 nearest neighbor from the starting point
            return sorted(range(len(order_pts)),
                          key=lambda k: abs(order_pts[k][0]-start_xy[0]) + abs(order_pts[k][1]-start_xy[1]))

        # Read route: Remove depot, convert to index sequence of original order_pts
        idx = routing.Start(0)
        order = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                order.append(node - 1)
            idx = sol.Value(routing.NextVar(idx))
        return order

    route_idx = _solve_tsp(pkg_pts)
    route = [pkgs[i] for i in route_idx]   # Fixed visiting sequence (slice of np.array), clear packages point by point
    route_ptr = 0

    # ---- force env._nearest_package to be the TSP order ----
    orig_nearest = env._nearest_package
    def _forced_nearest():
        nonlocal route_ptr
        if route_ptr < len(route):
            return route[route_ptr]
        return None
    env._nearest_package = _forced_nearest

    try:
        for _ in range(STEPS_PER_EP):
            action = 0  # Always package delivery
            obs, r, done, info, mask = env.step(action)
            total += r

            # This step may reach multiple package points at the same coordinates at once (theoretically allowed), use delivered_step to advance pointer
            delivered_cnt = int(info.get("delivered_step", 0))
            if delivered_cnt > 0:
                route_ptr = min(route_ptr + delivered_cnt, len(route))

            if done:
                terminal_time = float(info.get("terminal_time_min", env.t))
                ended_reason  = str(info.get("ended_reason", env._ended_reason or "unknown"))
                t_rides = float(info.get("time_rides_min", 0.0))
                t_delivery = float(info.get("time_delivery_min", 0.0))
                break

        if terminal_time is None:
            terminal_time = float(env.t)
            t_rides = float(getattr(env, "time_rides_min", 0.0))
            t_delivery = float(getattr(env, "time_delivery_min", 0.0))
    finally:
        # Restore environment method
        env._nearest_package = orig_nearest

    revenue_rate = total / terminal_time if terminal_time > 0 else 0.0
    return total, terminal_time, revenue_rate, ended_reason, 0, t_rides, t_delivery


