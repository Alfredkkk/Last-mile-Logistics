def _baseline_switching(env: CoModalEnv,
                        switch_grace_steps: Optional[int] = None,
                        seed: Optional[int] = None,
                        *, use_voronoi: bool = False
                        ) -> Tuple[float, float, float, str, int, float, float]:
    """Shared switching runner with package-first reservations and fixed windows.

    Keep the one-step pre-grace and the intentional post-delivery window. Each
    window belongs to one package: its screening point stays fixed even while
    an unmatched vehicle moves toward the next package. Completing another
    package starts that package's window. A pre-grace acceptance reserves the
    request and locks delivery to this window's package before pickup begins.
    """
    env.reset(seed=seed)
    gs = SWITCH_GRACE_STEPS if switch_grace_steps is None else int(switch_grace_steps)
    window_steps = max(1, gs)
    grace_left = 0
    window_package = None  # Stable global package index, including after delivery.
    reserved_package = None
    total = 0.0
    original_nearest = env._nearest_package

    def delivery_target():
        if reserved_package is not None and not env.pkg_delivered[reserved_package]:
            return env.pkg_pos_all[reserved_package].copy()
        return original_nearest()

    try:
        env._nearest_package = delivery_target
        for _ in range(STEPS_PER_EP):
            # A reservation survives display TTL and must be served after its
            # package is complete. No further screening occurs while committed.
            if (env.pending_ride is not None and reserved_package is not None
                    and env.pkg_delivered[reserved_package]):
                env._start_reserved_ride()
                reserved_package = None

            busy = env.to_pickup is not None or env.with_passenger
            if not busy and env.pending_ride is None:
                if grace_left == 0:
                    target = original_nearest()
                    if target is not None and manhattan(env.pos, target) <= env.v * env.dt:
                        window_package = min(
                            env.pkg_remaining_idx,
                            key=lambda i: manhattan(env.pkg_pos_all[i], target))
                        grace_left = window_steps

                if grace_left > 0:
                    reference = env.pkg_pos_all[window_package].copy()
                    visible = env._visible_rides(reference_point=reference)
                    chosen = None
                    # Visibility already orders pickups by distance from this
                    # fixed reference and applies the existing top-K/TTL rules.
                    for request in visible:
                        if use_voronoi:
                            owner = env.voronoi_owner_idx(request.dropoff)
                            # The current package (and coincident duplicates)
                            # will be delivered before this passenger is served.
                            if (env.pkg_delivered[owner] or
                                    manhattan(env.pkg_pos_all[owner], reference) < 1e-6):
                                continue
                        elif (HEUR_SOFT_PICK_CAP is not None and
                              manhattan(reference, request.pickup) > HEUR_SOFT_PICK_CAP):
                            continue
                        chosen = request
                        break

                    defer_pickup = not bool(env.pkg_delivered[window_package])
                    # Commit the request object, not its index: env.step(i)
                    # would resolve a different list around the next package.
                    if chosen is not None and env._accept_ride(chosen, defer_pickup=defer_pickup):
                        reserved_package = window_package if defer_pickup else None
                        grace_left = 0
                        window_package = None
                    else:
                        grace_left -= 1

            remaining_before = list(env.pkg_remaining_idx)
            # Busy: finish the passenger trip. Reserved: finish its locked
            # package. Unmatched: keep delivering, never wait in place.
            _, reward, done, info, _ = env.step(0)
            total += reward
            if done:
                break  # In particular, do not start a pickup after the horizon.

            if (info.get("delivered_step", 0) > 0 and env.pending_ride is None
                    and env.to_pickup is None and not env.with_passenger):
                delivered_now = [i for i in remaining_before if env.pkg_delivered[i]]
                if delivered_now:
                    window_package = min(
                        delivered_now, key=lambda i: manhattan(env.pos, env.pkg_pos_all[i]))
                    grace_left = window_steps
            # No proximity/window bookkeeping during pickup or passenger travel.
            # An actual delivery at a coincident dropoff can open a window above.
    finally:
        env._nearest_package = original_nearest

    terminal_time = float(env.t)
    revenue_rate = total / terminal_time if terminal_time > 0 else 0.0
    return (total, terminal_time, revenue_rate, env._ended_reason or "unknown",
            int(env.accepted_rides), float(env.time_rides_min), float(env.time_delivery_min))


def baseline_nearby_rule(env: CoModalEnv,
                         pickup_alpha=R_PICK_ALPHA,
                         drop_bias=0.5,
                         switch_grace_steps: Optional[int] = None,
                         seed: Optional[int] = None
                         ) -> Tuple[float, float, float, str, int, float, float]:
    """Switching without a passenger-destination restriction.

    pickup_alpha/drop_bias are retained legacy arguments. Screening uses
    env.r_pick around the window's package; HEUR_SOFT_PICK_CAP is optional.
    """
    return _baseline_switching(env, switch_grace_steps, seed, use_voronoi=False)


def baseline_nearby_rule_voronoi(env: CoModalEnv,
                                 pickup_alpha=R_PICK_ALPHA,
                                 drop_bias=0.5,
                                 switch_grace_steps: Optional[int] = None,
                                 seed: Optional[int] = None
                                 ) -> Tuple[float, float, float, str, int, float, float]:
    """Switching with dropoffs restricted to still-unserved L1 Voronoi cells.

    Shares HEUR's fixed windows and package-first reservations. Pre-grace
    screening excludes the package that will finish before passenger service.
    pickup_alpha/drop_bias remain legacy arguments; env.r_pick sets the radius.
    """
    return _baseline_switching(env, switch_grace_steps, seed, use_voronoi=True)
