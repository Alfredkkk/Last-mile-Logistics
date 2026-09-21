"""Source-only notebook edits for approved B2 observation/B4 convention changes."""
from edit_notebooks import edit_sources


def once(code, old, new):
    assert code.count(old) == 1, old
    return code.replace(old, new, 1)


ACCEPTED_FEATURES = '''    def _accepted_ride_features(self) -> List[float]:
        """Pending flag and current pickup/dropoff targets relative to the vehicle.

        Pickup features become zero after boarding; all targets clear at dropoff.
        A FOUR_ZONE commitment exposes known endpoints while delivery continues.
        Existing pickup/passenger flags distinguish active zero-distance targets.
        """
        pending = self.pending_ride
        pickup = pending.pickup if pending is not None else self.to_pickup
        dropoff = pending.dropoff if pending is not None else self.drop_target
        features = [float(pending is not None)]
        for target in (pickup, dropoff):
            if target is None:
                features.extend([0.0, 0.0, 0.0])
            else:
                relative = (target - self.pos) / self.R
                features.extend([float(relative[0]), float(relative[1]),
                                 manhattan(self.pos, target) / self.R])
        return features

'''


def transform(code):
    if 'LEGACY_GEOMETRY = {"R": 5.5' in code:
        code += '\n# R_PICK_ALPHA and ALPHA_EFFECTIVE use code units; paper alpha = code alpha / sqrt(2).\n'
    if "# City / motion" in code:
        code = once(code, 'R = 5.5                 # half "L1 radius" of diamond region; feasible points satisfy |x|+|y| <= R',
                    'R = 5.5                 # Code L1 radius; same-region paper edge length R_paper = sqrt(2) * R')
        code = once(code, 'R_PICK_ALPHA = 0.5      # pickup visibility radius parameter r = alpha * R / sqrt(2) (L1-constraint approx)',
                    'R_PICK_ALPHA = 0.5      # Code alpha: r = alpha * R / sqrt(2); alpha_paper = alpha / sqrt(2)')
    if "class CoModalEnv:" in code:
        code = once(code, "        * up to K rides visible: for each, (dx_pick/R, dy_pick/R, l1_pick/R, dx_drop/R, dy_drop/R, l1_trip/R)",
                    "        * up to K rides visible: (dx_pick/R, dy_pick/R, l1_pick/R, dx_drop/R, dy_drop/R, l1_trip/R, ttl_fraction)\n"
                    "        * accepted trip: pending flag plus current pickup/dropoff (dx/R, dy/R, l1/R)\n"
                    "      Observation remains partial: distant packages and hidden buffered requests are omitted.")
        code = once(code, "        # max pickup radius (L1) following r = alpha * R / sqrt(2); we keep L1 constraint",
                    "        # Preserve code radius: paper alpha = r_pick_alpha / sqrt(2), R_paper = sqrt(2) * R.")
        code = once(code, "    def _get_obs(self) -> Tuple[np.ndarray, np.ndarray]:", ACCEPTED_FEATURES +
                    "    def _get_obs(self) -> Tuple[np.ndarray, np.ndarray]:")
        code = once(code, "        ride_feats = []\n", "        ride_feats = []\n"
                    "        # Match by request identity: visibility sorting must not reorder TTLs incorrectly.\n"
                    "        ttl_by_request = {id(req): ttl for req, ttl in self.ride_buffer}\n")
        code = once(code, "                               reld[0], reld[1], r.trip_len / self.R])",
                    "                               reld[0], reld[1], r.trip_len / self.R,\n"
                    "                               ttl_by_request[id(r)] / float(self.ride_ttl_steps)])")
        code = once(code, "        while len(ride_feats) < 6 * self.max_visible:",
                    "        while len(ride_feats) < 7 * self.max_visible:")
        code = once(code, "        obs = np.array(core + pack_feats + ride_feats, dtype=np.float32)",
                    "        obs = np.array(core + pack_feats + ride_feats + self._accepted_ride_features(), dtype=np.float32)")
        code = once(code, " + 3 * self.k_pack + 6 * self.max_visible", " + 3 * self.k_pack + 7 * self.max_visible + 7")
        code = once(code, " + 3*K + 6*MAX_VISIBLE", " + 3*K + 7*MAX_VISIBLE + 7 accepted-trip features")
    if "def run_param_sweep(" in code:
        code = once(code, "TRAIN_UPDATES=train_updates_per_combo, REPORT_UNSCALED=REPORT_UNSCALED,",
                    "TRAIN_UPDATES=train_updates_per_combo, REPORT_UNSCALED=REPORT_UNSCALED, OBS_DIM=env.obs_dim,")
    if "def plot_single_param_trend(" in code:
        code = once(code, "    plt.xlabel(vary_key)\n", "    plt.xlabel('Effective alpha (code scale)' if vary_key == 'ALPHA_EFFECTIVE' else vary_key)\n")
    return code


if __name__ == "__main__":
    for filename in ("experiment.ipynb", "NonStationary/experiment2.ipynb", "analysis.ipynb", "NonStationary/analysis2.ipynb"):
        edit_sources(filename, transform)
