from pathlib import Path

from edit_notebooks import edit_sources, replace_function


RIDE_METHODS = '''    def _accept_ride(self, chosen: RideReq, defer_pickup: bool = False) -> bool:
        """Commit a buffered request once; TTL applies only before acceptance.

        The caller selects a visible, policy-eligible request. FOUR_ZONE may
        defer pickup while finishing its current delivery zone. Other policies
        retain immediate pickup through step(action).
        """
        if self.to_pickup is not None or self.with_passenger or self.pending_ride is not None:
            return False
        for i, (req, ttl) in enumerate(self.ride_buffer):
            if req is chosen and ttl > 0:
                self.ride_buffer.pop(i)
                self.accepted_rides += 1
                if defer_pickup:
                    self.pending_ride = chosen
                else:
                    self.to_pickup = chosen.pickup.copy()
                    self.drop_target = chosen.dropoff.copy()
                return True
        return False

    def _start_reserved_ride(self) -> bool:
        """Begin pickup of an accepted request without a second TTL check/count."""
        if self.pending_ride is None or self.to_pickup is not None or self.with_passenger:
            return False
        chosen = self.pending_ride
        self.pending_ride = None
        self.to_pickup = chosen.pickup.copy()
        self.drop_target = chosen.dropoff.copy()
        return True

'''


def fix_experiment(code):
    # Both CoModalEnv and FixedPackageEnv define their own reset method.
    code = code.replace(
        '        self.drop_target: Optional[np.ndarray] = None\n',
        '        self.drop_target: Optional[np.ndarray] = None\n'
        '        self.pending_ride: Optional[RideReq] = None  # FOUR_ZONE: accepted, delivery before pickup\n')
    if 'class CoModalEnv:' in code:
        code = code.replace('    def _visible_rides(self) -> List[RideReq]:\n'
                            '        # Show rides whose pickup is within r_pick (L1) of the delivery target and reachable before TTL expires.\n'
                            '        ref = self._visibility_reference_point()\n',
                            '    def _visible_rides(self, reference_point: Optional[np.ndarray] = None,\n'
                            '                       pickup_radius: Optional[float] = None) -> List[RideReq]:\n'
                            '        # FOUR_ZONE supplies its final zone delivery point; other policies use the current target.\n'
                            '        ref = self._visibility_reference_point() if reference_point is None else reference_point\n'
                            '        radius = self.r_pick if pickup_radius is None else pickup_radius\n')
        code = code.replace('            if dist_to_pickup > self.r_pick:\n', '            if dist_to_pickup > radius:\n')
        code = code.replace('            # 2) TTL + ETA filtering: must be able to reach pickup from that reference before TTL expires.\n',
                            '            # 2) Retained reference-point ETA filter; TTL is a display lifetime, not a pickup deadline.\n')
        marker = '    def _nearest_package(self) -> Optional[np.ndarray]:\n'
        assert code.count(marker) == 1
        code = code.replace(marker, RIDE_METHODS + marker)
        old = '''            if action > 0 and action <= len(visible):
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
                self.accepted_rides += 1
'''
        assert code.count(old) == 1
        code = code.replace(old, '''            if (0 < action <= len(visible) and
                    self._accept_ride(visible[action - 1])):
''')
        code = code.replace('        done_packages = (len(self.packages) == 0) and (not self.with_passenger) and (self.to_pickup is None)',
                            '        done_packages = ((len(self.packages) == 0) and (not self.with_passenger)\n'
                            '                         and (self.to_pickup is None) and (self.pending_ride is None))')
        code = code.replace('        for i in range(len(visible)):\n            mask[1 + i] = 1.0\n',
                            '        if self.pending_ride is None:\n'
                            '            for i in range(len(visible)):\n                mask[1 + i] = 1.0\n')
    if 'def baseline_pure_ortools(' in code:
        # The artificial return arc costs zero: fixed start, free final package.
        old = '            return dist[i][j]\n\n        cb_idx = routing.RegisterTransitCallback(cb)'
        assert code.count(old) == 1
        code = code.replace(old, '            return 0 if routing.IsEnd(to_idx) else dist[i][j]\n\n        cb_idx = routing.RegisterTransitCallback(cb)')
        code = code.replace('    Pure delivery baseline: Use OR-Tools to solve an L1-TSP for all package points in the current episode,',
                            '    Pure delivery baseline: Use OR-Tools to solve an open L1 route for all package points,')
        code = code.replace('    # ---- Solve TSP once: depot=start, nodes=packages ----',
                            '    # ---- Open route: keep departure costs, set the artificial return arc to zero ----')
    if 'def solve_zone_tsp_L1(' in code:
        code = code.replace('    # OR-Tools requires an integer matrix; here we use an integer approximation by scaling by 1000, or simply rounding.\n'
                            '    return int(round(abs(a[0]-b[0]) + abs(a[1]-b[1])))',
                            '    # Match PURE_OR precision: 1000 integer units per distance unit.\n'
                            '    return int(round(1000 * (abs(a[0]-b[0]) + abs(a[1]-b[1]))))')
        old = '        return dist[i][j]\n\n    transit_cb_idx = routing.RegisterTransitCallback(distance_callback)'
        assert code.count(old) == 1
        code = code.replace(old, '        return 0 if routing.IsEnd(to_index) else dist[i][j]\n\n    transit_cb_idx = routing.RegisterTransitCallback(distance_callback)')
        code = code.replace('    # Allow not returning to the depot; by default, it will find a path starting from the depot',
                            '    # Only the artificial return arc is free; the start-to-first-package cost is retained.')
        code = replace_function(code, 'baseline_four_zone', Path(__file__).with_name('four_zone.py').read_text(encoding='utf-8'))
    return code


if __name__ == '__main__':
    for filename in ('experiment.ipynb', 'NonStationary/experiment2.ipynb'):
        edit_sources(filename, fix_experiment)
