def baseline_four_zone(env: CoModalEnv,
                       r_pick_alpha: Optional[float] = None,
                       screen_window_min: float = 10.0,
                       seed: Optional[int] = None
                       ) -> Tuple[float, float, float, str, int, float, float]:
    """Four-zone policy with advance acceptance and delivery before pickup.

    Follow an open delivery route within each zone. During its final t0 minutes,
    screen from the route's final delivery point and reserve at most one ride to
    another unserved zone. Finish this zone, serve that ride, then deliver in its
    destination zone. Without a ride, move to the nearest unserved zone.

    TTL, the additional reference-point ETA filter, visibility cap, and the
    project's radius convention remain in force. None uses the environment's
    pickup radius; an explicit r_pick_alpha overrides it for this baseline.
    """
    if not math.isfinite(screen_window_min) or screen_window_min < 0:
        raise ValueError("screen_window_min must be finite and nonnegative")
    if env.v <= 0 or env.dt <= 0:
        raise ValueError("Four-zone routing requires positive speed and time step")
    env.reset(seed=seed)
    total = 0.0
    r_pick = env.r_pick if r_pick_alpha is None else r_pick_alpha * env.R / math.sqrt(2.0)
    package_zones = [l1_zone_id(p) for p in env.pkg_pos_all]
    current_zone = None
    route = []  # Global package indices; delivered duplicates can be skipped safely.
    screening_ref = env.pos.copy()
    zone_finish_time = math.inf
    ride_destination = None
    phase = "delivery"

    def zone_indices(zid):
        return [i for i in env.pkg_remaining_idx if package_zones[i] == zid]

    def unserved_zones():
        return {package_zones[i] for i in env.pkg_remaining_idx}

    def pick_next_zone():
        candidates = sorted(unserved_zones())
        if not candidates:
            return None
        return min(candidates, key=lambda z: min(
            manhattan(env.pos, env.pkg_pos_all[i]) for i in zone_indices(z)))

    def predict_zone_finish():
        """Dry-run delivery motion once per zone, without changing the environment.

        Include per-step rounding and the existing boundary projection, so t0
        refers to the actual simulated completion time, not just distance / v.
        A zone that cannot finish within the horizon plus t0 cannot open a window
        during this episode; bound prediction work and return infinity for it.
        """
        pos = env.pos.copy()
        remaining = list(route)
        max_steps = int(math.ceil((max(0.0, HORIZON_MIN - env.t) + screen_window_min) / env.dt)) + 1
        for step_count in range(1, max_steps + 1):
            new_pos, _, _ = step_towards(pos, env.pkg_pos_all[remaining[0]], env.v * env.dt)
            pos = project_to_diamond(new_pos, env.R)
            remaining = [i for i in remaining if manhattan(pos, env.pkg_pos_all[i]) >= 1e-6]
            if not remaining:
                # Later route entries may already have been delivered at duplicate
                # coordinates or along an earlier leg. Use the actual final point.
                return env.t + step_count * env.dt, pos.copy()
        return math.inf, env.pkg_pos_all[route[-1]].copy()

    def enter_zone(zid):
        nonlocal current_zone, route, screening_ref, zone_finish_time
        current_zone = zid
        indices = zone_indices(zid) if zid is not None else []
        points = [tuple(map(float, env.pkg_pos_all[i])) for i in indices]
        order = solve_zone_tsp_L1(points, tuple(map(float, env.pos))) if points else []
        route = [indices[k] for k in order]
        zone_finish_time, screening_ref = predict_zone_finish() if route else (env.t, env.pos.copy())

    def forced_target():
        return env.pkg_pos_all[route[0]].copy() if route else None

    original_nearest = env._nearest_package
    try:
        env._nearest_package = forced_target
        enter_zone(pick_next_zone())
        for _ in range(STEPS_PER_EP):
            # Complete the accepted trip before selecting any more requests.
            if phase == "ride" and env.to_pickup is None and not env.with_passenger:
                destination = ride_destination
                ride_destination = None
                phase = "delivery"
                enter_zone(destination if zone_indices(destination) else pick_next_zone())

            while phase == "delivery":
                route = [i for i in route if not env.pkg_delivered[i]]
                destinations = unserved_zones() - {current_zone}
                # Include the completion instant; never wait after the zone is done.
                # For a zone shorter than t0, screening begins when service starts.
                if (env.pending_ride is None and destinations and
                        zone_finish_time - env.t <= screen_window_min + 1e-9):
                    visible = env._visible_rides(reference_point=screening_ref, pickup_radius=r_pick)
                    for req in visible:
                        destination = l1_zone_id(req.dropoff)
                        if destination in destinations and env._accept_ride(req, defer_pickup=True):
                            ride_destination = destination
                            break

                if route:
                    break  # Continue delivering even if a ride has been accepted.
                if env.pending_ride is not None:
                    env._start_reserved_ride()
                    phase = "ride"
                    break
                next_zone = pick_next_zone()
                if next_zone is None:
                    break
                enter_zone(next_zone)

            _, reward, done, _, _ = env.step(0)
            total += reward
            if done:
                break
    finally:
        env._nearest_package = original_nearest

    terminal_time = float(env.t)
    revenue_rate = total / terminal_time if terminal_time > 0 else 0.0
    return (total, terminal_time, revenue_rate, env._ended_reason or "unknown",
            int(env.accepted_rides), float(env.time_rides_min), float(env.time_delivery_min))
