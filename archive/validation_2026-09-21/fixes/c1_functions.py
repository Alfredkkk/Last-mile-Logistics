def train_policy_brief(env: CoModalEnv,
                       updates: int,
                       *,
                       eval_every: int = 10,
                       heur_grace_steps: Optional[int] = None,
                       log_path: Optional[str] = None,
                       combo_meta: Optional[dict] = None,
                       checkpoint_dir=None, checkpoint_every: int = 5,
                       resume: bool = False, show_progress: bool = True,
                       progress_callback=None):
    """Train/evaluate with optional complete-update checkpoints and progress.

    Resume requires the same configuration and total update count. Checkpoints
    own a separate training_log.csv; standalone calls without a checkpoint_dir
    retain the original logging API. Returns the final policy and its metrics.
    """
    from training_persistence import TrainingCheckpoint, TrainingProgress, training_settings
    if int(updates) != updates or updates < 0:
        raise ValueError("updates must be a nonnegative integer")
    if int(checkpoint_every) != checkpoint_every or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer")
    if resume and checkpoint_dir is None:
        raise ValueError("resume requires checkpoint_dir")
    updates = int(updates)
    checkpoint = None
    if checkpoint_dir is not None:
        expected_log = Path(checkpoint_dir).resolve() / "training_log.csv"
        if log_path is not None and Path(log_path).resolve() != expected_log:
            raise ValueError("Checkpointed training must use its own directory's training_log.csv")
        settings = dict(training_settings(globals()), updates=updates, eval_every=eval_every,
                        heur_grace_steps=heur_grace_steps)
        checkpoint = TrainingCheckpoint(checkpoint_dir, env, settings, combo_meta, resume=resume)
        log_path = checkpoint.log_path

    policy = ActorCritic(env.obs_dim, env.act_dim).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    policy.train()
    completed, last_eval, last_metrics = 0, None, None
    rollout_state = RolloutState()
    if resume:
        completed, last_eval, last_metrics, rollout_state = checkpoint.load(
            policy, optimizer, env, globals(), RideReq)
    saved_update = completed if resume else None
    progress = TrainingProgress(updates, initial=completed, enabled=show_progress,
                                path=checkpoint.directory / "progress.json" if checkpoint else None,
                                callback=progress_callback)

    def report(phase):
        progress.report(completed, phase, saved_update=saved_update)

    def save(final=False):
        nonlocal saved_update
        if checkpoint:
            checkpoint.save(policy, optimizer, env, rollout_state, globals(), update=completed,
                            last_eval=last_eval, metrics=last_metrics, final=final)
            saved_update = completed

    def evaluate_and_log():
        nonlocal last_eval, last_metrics
        report("evaluating")
        metrics = evaluate_all(env, policy, heur_grace_steps=heur_grace_steps)
        if log_path:
            drl = metrics["drl"]
            eval_ep_stats = [{'reward': drl['avg_reward'], 'rate': drl['avg_rate'], 'ep_rate': drl['avg_ep_rate'], 'terminal_time_min': drl['avg_t'], 'steps': float('nan')}]
            append_training_log(log_path, completed, 'eval', eval_ep_stats, None, None,
                                (combo_meta or {}).copy(), values_are_unscaled=REPORT_UNSCALED)
        last_eval, last_metrics = completed, metrics
        save(final=completed == updates)
        drl = metrics["drl"]
        print(f"[Eval {completed}/{updates}] DRL rate={drl['avg_rate']:.4f}/min, "
              f"finish_rate={drl['finish_rate']:.2f}")

    try:
        if not resume:
            save()  # Update zero is recoverable even if the first batch fails.
        report("resumed" if resume else "starting")
        # A failure during evaluation must not repeat an already committed PPO update.
        if last_eval != completed and (completed == updates or
                                      (completed > 0 and eval_every and completed % eval_every == 0)):
            evaluate_and_log()
        for upd in range(completed + 1, updates + 1):
            report("sampling_and_ppo")
            batch, ep_stats, step_avg_reward, step_avg_rate = collect_rollout(env, policy, PPO_STEPS, rollout_state)
            for _ in range(PPO_EPOCHS):
                for mb in make_minibatches(batch, PPO_MINI_BATCH):
                    ppo_update(policy, optimizer, mb, CLIP_EPS)
            completed = upd
            if log_path:
                append_training_log(log_path, upd, 'train', ep_stats, step_avg_reward, step_avg_rate, combo_meta or {})
            if checkpoint and (upd % checkpoint_every == 0 or upd == updates):
                save()  # Keep trained weights even if the following evaluation fails.
            if (eval_every and upd % eval_every == 0) or upd == updates:
                evaluate_and_log()
            report("update_complete")
        save(final=True)  # Also repairs a final-file write interrupted after latest.pt was saved.
        report("complete")
        return policy, last_metrics
    except BaseException as error:
        # Never checkpoint an interrupted rollout or partly updated optimizer.
        report("interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
        raise
    finally:
        progress.close()


def run_param_sweep(
    LAMBDA_list, R_PICK_ALPHA_list, GAMMA_PACK_list, RIDE_TTL_MINUTES_list,
    MAX_VISIBLE_RIDES_list, SWITCH_GRACE_STEPS_list, RT_list, *,
    train_updates_per_combo: int = 0, csv_path=None, seed_offset: int = 0,
    train_seeds=None, checkpoint_every: int = 5, resume_dir=None,
    show_progress: bool = True,
):
    """Persist each effective setting/seed and resume from a saved sweep directory.

    New runs get independent artifact folders. resume_dir skips committed groups
    and continues the unfinished group from its latest complete PPO update.
    train_seeds are absolute paired seeds and cannot be combined with seed_offset.
    """
    from training_persistence import SweepStore, training_settings
    if csv_path is None:
        csv_path = "param_sweep_results.csv"
    if int(train_updates_per_combo) != train_updates_per_combo or train_updates_per_combo < 0:
        raise ValueError("train_updates_per_combo must be a nonnegative integer")
    if int(checkpoint_every) != checkpoint_every or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer")
    if train_seeds is not None and seed_offset != 0:
        raise ValueError("Use explicit train_seeds or seed_offset, not both")
    seeds = [SEED + seed_offset] if train_seeds is None else list(train_seeds)
    plan = build_sweep_plan(
        LAMBDA_list, R_PICK_ALPHA_list, GAMMA_PACK_list, RIDE_TTL_MINUTES_list,
        MAX_VISIBLE_RIDES_list, SWITCH_GRACE_STEPS_list, RT_list,
        R=R, v=V, dt=DT, train_seeds=seeds)
    if not plan:
        raise ValueError("The sweep grid is empty")
    settings = dict(training_settings(globals()), RP=RP, updates=train_updates_per_combo,
                    environment=CoModalEnv.__name__)
    store = SweepStore(csv_path, plan, settings, resume_dir=resume_dir)
    print(f"[GRID] {len(plan)} groups; {len(store.completed)} already saved; seeds={seeds}")
    print(f"[GRID] Run folder (use as resume_dir): {store.directory}")
    try:
        for setting in plan:
            combo_id = setting['combo_id']
            if combo_id in store.completed:
                continue
            print(f"[GRID {combo_id}/{len(plan)}] {len(store.completed)} saved; "
                  f"alpha={setting['R_PICK_ALPHA']}, seed={setting['train_seed']}")
            store.status("training", combo_id=combo_id)
            seed = setting['train_seed']
            set_global_seeds(seed)
            env = CoModalEnv(R=R, v=V, dt=DT, rp=RP,
                            lam=setting['LAMBDA'], gamma_pack=setting['GAMMA_PACK'],
                            r_pick_alpha=setting['R_PICK_ALPHA'],
                            ride_ttl_minutes=setting['RIDE_TTL_MIN'],
                            max_visible=setting['MAX_VISIBLE_RIDES'], rt=setting['RT'], seed=seed)
            meta = dict(setting, RP=RP, HORIZON_MIN=HORIZON_MIN,
                        TRAIN_UPDATES=train_updates_per_combo, REPORT_UNSCALED=REPORT_UNSCALED, OBS_DIM=env.obs_dim,
                        DEMAND_PROFILE=(json.dumps(env.hourly_multiplier.tolist())
                                        if hasattr(env, 'hourly_multiplier') else 'stationary'))
            directory = store.combo_dir(combo_id)
            directory.mkdir(parents=True, exist_ok=True)
            policy, metrics = train_policy_brief(
                env, updates=train_updates_per_combo, eval_every=5,
                heur_grace_steps=setting['SWITCH_GRACE_STEPS'],
                log_path=directory / "training_log.csv", combo_meta=meta,
                checkpoint_dir=directory, checkpoint_every=checkpoint_every,
                resume=(directory / "latest.pt").is_file(), show_progress=show_progress,
                progress_callback=lambda status: store.status("training", combo_id=combo_id, training=status))
            rows = []
            for algo_key in ("drl", "heur", "heur_vor", "four_zone", "pure", "pure_or"):
                m = metrics[algo_key]
                rows.append(dict(meta, algo=algo_key.upper(),
                                 reward=float(m['avg_reward']), rate=float(m['avg_rate']),
                                 ep_rate=float(m['avg_ep_rate']), terminal_time=float(m['avg_t']),
                                 avg_finish_time=float(m['avg_finish_time']), finish_rate=float(m['finish_rate']),
                                 accepted=float(m.get('avg_acc', float('nan'))),
                                 time_rides=float(m.get('avg_time_rides', float('nan'))),
                                 time_delivery=float(m.get('avg_time_delivery', float('nan')))))
            store.commit(combo_id, rows)
        store.status("complete")
        df = pd.DataFrame(store.rows)
        df.attrs["run_directory"] = str(store.directory)
        print(f"[GRID] Saved {len(df)} rows to {store.csv_path}")
        return df
    except BaseException as error:
        store.status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error_type=type(error).__name__)
        raise
