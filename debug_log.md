# Debug Log

> Latest implementation update: **2026-09-21 Paths, Run Identity and Archive Cleanup** below. All four notebooks use scenario-rooted paths; convergence selects one identified new run/group. run_id persists through logs/results/checkpoints/resume and separates independent-run summaries. The 53-test suite and recent debugging material are archived under archive/validation_2026-09-21 after validation. Five historical training logs remain in archive/training_logs. Other open items and accepted/deferred modeling decisions still apply.

## 2026-05-31 Notebook and Experiment Logic Review

### Goal

Record the issues found in the May 31, 2026 review of the stationary and non-stationary last-mile logistics notebooks, then track the first debug pass for issues 1-7.

### User Decisions

- Create a separate root-level `debug_log.md` instead of appending to `progress_log.md`.
- Apply first-round code fixes to both `experiment.ipynb` and `NonStationary/experiment2.ipynb`.
- Keep the current code geometry convention: diamond feasible region `|x| + |y| <= R` with area `2 * R^2`.
- Align documentation and comments with the existing `2 * R^2` convention.
- Treat `main()` as an unused/obsolete training entry and disable it rather than maintaining it as a supported path.
- Handle the remaining model-alignment and analysis issues in later stages.
- Treat issue 10 as an intentional modified switching heuristic: the pre-grace window is kept because it improved training/evaluation behavior relative to the stricter paper timing.
- Fix issue 11 after advisor confirmation: ride visibility is now screened from the package delivery target rather than the vehicle's current position.
- Treat issue 12 as out of scope: the project does not need to restrict parameter ranges to the paper's numerical figures.
- For issue 19, use `rate` / `avg_rate` as the main time-weighted aggregate metric and keep `ep_rate` / `avg_ep_rate` as a diagnostic mean of per-episode rates.
- For issues 20-21, replace fragile path strings with notebook-local path resolvers that locate the project root and non-stationary results directory.
- For issue 22, keep `time_frac` and add cyclic hour-of-day observation features so the policy can learn the 24-hour demand cycle.
- Treat issue 23 as deferred: fixed start time may matter for robustness studies, but it is not a current blocking bug.
- Treat issue 24 as deferred: the simulation keeps the paper-style abstract diamond region and does not attempt to spatially map NYC geography into the model.
- For issues 25-26, align README and notebook comments with the active `run_param_sweep()` workflow and the code geometry convention `area = 2 * R^2`.
- For issue 27, keep the legacy baseline parameters but explicitly document that they are not active in the current switching heuristic.
- Treat issue 28 as deferred: keep the current movement implementation for now, but retain L1 geodesic interpolation as the preferred future fix if boundary projection becomes material.
- Current training-log pipeline uses fresh fixed-schema `training_log.csv` files directly. Legacy training logs were archived under `archive/training_logs/`.
- Treat the pre-fix active CSV/PNG files as expected historical outputs until the post-debug experiments are rerun; their current age/schema is not a code blocker.
- Continue reporting revenue rate for horizon-truncated episodes without adding an unfinished-package penalty; incomplete delivery remains visible through `finish_rate`.
- Defer issue 29 for future research: investigate whether PPO's discounted-return objective contributes to low package completion rates or differs materially from the evaluated revenue-rate objective.
- On 2026-09-20, define TTL as the display/availability lifetime of an unaccepted ride request and keep the current implementation. It is not an actual pickup deadline; CODE_REVIEW_2026-09-05.md B3 no longer requires a pickup-deadline fix under this definition.
- On 2026-09-21, complete issue 17 first, then fix all four FOUR_ZONE issue groups (A1-A4). Align its screening/advance-acceptance process with the paper. Correct the shared open-route objective in PURE_OR as well.
- On 2026-09-21, also implement the agreed A5/A6/A8/A9 fixes: classify time by the executed activity, filter scatter plots to the intended package densities, evaluate/log the final model once, and independently copy the non-stationary hourly profile into evaluation environments.
- On 2026-09-21, implement B1 by continuing unfinished episodes across PPO updates, and B5 by dynamically grouping equivalent alpha settings and explicitly tracking training seeds. Keep display TTL and the additional ETA visibility filter; changes to parameters must automatically change equivalence grouping.
- On 2026-09-21, implement B2 by exposing accepted-trip pickup/dropoff relative positions and L1 distances plus visible-request remaining TTL; retain nearest-10 package compression. For B4 choose option 1: keep the current physical radius and explicitly map `R_paper=sqrt(2)*R_code`, `alpha_paper=alpha_code/sqrt(2)` in code metadata and documentation.
- On 2026-09-21, implement per-group result saving, model/training checkpoints and visible training progress. Explain the missing-run-identifier issue; do not treat this as approval for a general historical-log or cross-run-analysis redesign. Separate run/group artifact folders are part of safe checkpoint recovery.
- Subsequently on 2026-09-21, the user authorized connecting analysis to the new log directories, auditing/fixing directory use in all four notebooks, implementing run_id, archiving old logs, and archiving/removing recent tests and temporary debugging files after all changes and tests are complete.

### First-Round Scope (Historical)

- In scope now: issues 1, 3, 4, 5, 6, and 7.
- Completed in this pass: issue 2 in `analysis.ipynb`.
- Completed in the 2026-06-01 follow-up pass: issues 13, 14, 15, and 16.
- Not in scope now: full historical CSV/PNG regeneration, broad README cleanup beyond issue-specific metric definitions, Uber data path cleanup, or full paper-level policy redesign beyond the concrete bugs below.

## Issue Inventory

### High Priority

1. `experiment.ipynb` and `NonStationary/experiment2.ipynb`: `main()` is stale and would crash because `collect_rollout()` now returns four values while `main()` still treats it as a single PPO batch. Status: fixed by disabling the obsolete entry; reconfirmed in both notebooks on 2026-09-19.
2. `analysis.ipynb`: `paper_axes()` contained stray CSV-reading code with undefined `path` / `r`, so later plotting cells could fail. Status: fixed on 2026-05-31 by removing the stray CSV block from `paper_axes()` while keeping `read_training_log(path)` separate.
3. `experiment.ipynb` and `NonStationary/experiment2.ipynb`: `baseline_nearby_rule_voronoi()` filters `visible` rides but converts the filtered candidate index directly into an action, so it can accept the wrong ride. Status: fixed by retaining original visible indices; reconfirmed on 2026-09-19.
4. `experiment.ipynb` and `NonStationary/experiment2.ipynb`: `baseline_four_zone()` computes a zone route but still lets `env.step(0)` use global nearest-package routing. Status: the original route-following bug was fixed and reconfirmed on 2026-09-19. The additional state-transition, screening, and route-cost problems (review A1-A4) were also fixed on 2026-09-21; see the implementation/validation entry below.
5. `experiment.ipynb` and `NonStationary/experiment2.ipynb`: `step()` samples new rides before resolving the action from the previous observation/mask, so ride action indices can refer to a different visible set than the policy observed. Status: fixed by appending new arrivals after action execution; action-index probes passed again on 2026-09-19.
6. `experiment.ipynb` and `NonStationary/experiment2.ipynb`: PPO `ActorCritic` uses dropout, so old and current log probabilities can differ even without parameter changes. Status: dropout removed from both current networks; reconfirmed on 2026-09-19.
7. `experiment.ipynb` and `NonStationary/experiment2.ipynb`: training rollouts are not fully reproducible because `CoModalEnv` initializes its local RNG without a seed. Status: environment seed support and seeded sweep construction implemented; seeded environment probes passed again on 2026-09-19. This does not establish bitwise reproducibility of GPU training or time-limited routing solvers.

### Paper Model and Experiment Definition

8. The code uses `|x| + |y| <= R` with area `2 * R^2`, while the paper defines edge-length `R` with area `R^2`. Status: resolved on 2026-06-01 by keeping the code convention and documenting it as the project geometry.
9. `readme.md` and code comments disagree about diamond area (`R^2` vs `2 * R^2`). Status: fixed on 2026-06-01 through the issue 25/26 README and notebook comment cleanup.
10. `baseline_nearby_rule()` uses a pre-grace window that may accept a ride before actually completing the package delivery, which is more aggressive than the paper's switching description. Status: accepted intentional modeling change; keep current behavior and document as a modified/pre-grace switching heuristic.
11. Ride visibility is computed from current vehicle position, not the final delivery location used in the paper's screening logic. Status: fixed on 2026-06-09 after advisor confirmation by adding a delivery-target visibility reference in both experiment notebooks and aligning heuristic ride selection with that reference.
12. Project sweeps use arrival-rate and TTL settings that are extensions, not a direct reproduction of the paper's Figure 7/8 calibration. Status: accepted; no action needed because the project is not constrained to the paper's plotted parameter range.

### Results and Analysis

13. `append_training_log()` double-descales eval rows because `evaluate_all()` already reports unscaled metrics. Status: fixed on 2026-06-01 by adding `values_are_unscaled` to `append_training_log()` and passing it for eval rows.
14. Training log schema is unstable because appended rows can have changing field names. Status: fixed on 2026-06-01 by adding fixed `TRAIN_LOG_COLUMNS` and writing all train/eval rows with that schema.
15. `analysis.ipynb` revenue-vs-n plots label `r_l=6` without filtering `RT == 6.0`. Status: fixed on 2026-06-01 by adding `RT_FILTER = 6.0` to the revenue-vs-n cells in both analysis notebooks.
16. `ratio_hvor_drl` uses best alpha for `HEUR_VOR` but average alpha for `DRL`, making the ratio definition inconsistent. Status: fixed on 2026-06-01 by comparing `HEUR_VOR` best alpha against `DRL` best alpha in both analysis notebooks.
17. "Optimal alpha" plots apply alpha optimization to pure delivery baselines, where alpha has no model meaning. Status: fixed completely on 2026-09-21. In addition to the June experiment-plot fixes, `best_alpha_series()` and `best_alpha_by_lambda_gamma()` in both analysis notebooks now average PURE/PURE_OR run rates across alpha. Ride policies retain selection by the best per-alpha mean. See review A7 and the regression tests.
18. Evaluation uses only `EVAL_EPISODES = 5`, so reported comparisons may have high Monte Carlo variance. Status: recommendation recorded on 2026-06-01. Keep `EVAL_EPISODES = 5` for quick/debug sweeps; use `20` for final full sweeps; use `30` for small final confirmation runs when runtime is manageable.
19. Aggregation uses mean of per-episode rates instead of `E[Reward] / E[Time]`, while the paper's objective is the latter. Status: fixed on 2026-06-01 by redefining the main `avg_rate` / CSV `rate` metric as `sum(reward) / sum(terminal_time)` and adding diagnostic `avg_ep_rate` / CSV `ep_rate` as `mean(reward_i / terminal_time_i)`.
29. PPO is trained with a discounted cumulative-reward objective, while final policy performance is evaluated using revenue rate. This objective difference may be one possible contributor to low package completion rates, but it has not been established as the cause. Status: deferred by user decision on 2026-08-07; keep the current PPO objective, reward function, horizon handling, and incomplete-episode revenue-rate reporting unchanged, and revisit this as a future research question.

### Metric Definitions Adopted for Issue 19

- `avg_rate` / CSV `rate`: primary time-weighted revenue rate, `sum_i reward_i / sum_i terminal_time_i`. This is the metric used by the main analysis plots.
- `avg_ep_rate` / CSV `ep_rate`: diagnostic mean of per-episode rates, `mean_i(reward_i / terminal_time_i)`. This remains useful for spotting episode-level variability but is not the primary comparison metric.

### Non-Stationary Scenario

20. `NonStationary/experiment2.ipynb` hard-codes a local absolute Uber data path. Status: fixed on 2026-06-01 by adding project-root path helpers and resolving Uber data from `UBER_PARQUET_DIR`, `../Uber_NYC`, `Uber_NYC`, or `NonStationary/Uber_NYC` without hard-coded user paths.
21. `NonStationary/analysis2.ipynb` uses relative `"Results/..."` paths that can point to the stationary results when run from the project root. Status: fixed on 2026-06-01 by routing non-stationary CSV/log reads and key figure outputs through `ns_results_path()` / `ns_path()`.
22. Non-stationary demand is represented only through `time_frac`, not explicit cyclic hour-of-day features. Status: fixed on 2026-06-01 by adding `hour_sin` and `hour_cos` to the `NonStationary/experiment2.ipynb` observation core and increasing `obs_dim` by 2.
23. Every non-stationary episode starts at `t = 0`, making results potentially start-time sensitive. Status: deferred by user decision on 2026-06-01; not considered urgent for the current debug pass.
24. The Uber NYC hourly profile is not spatially filtered to the Manhattan/diamond service region. Status: deferred by user decision on 2026-06-01; the project intentionally keeps the paper-style abstract diamond region and uses Uber NYC only for city-wide hour-of-day demand variation.

### Documentation and Code Hygiene

25. `readme.md` defaults are stale (`V`, `HORIZON_MIN`, `RT`, `RP`, and Quick Start do not match current code). Status: fixed on 2026-06-01 by rewriting Quick Start around `run_param_sweep()`, removing `main()` usage, and updating current defaults.
26. Environment comments still describe package count as `Poisson(gamma * R^2)` even though code uses `Poisson(gamma * 2R^2)`. Status: fixed on 2026-06-01 by aligning README and experiment notebook comments to the `2 * R^2` L1 diamond area convention.
27. `baseline_nearby_rule()` accepts `pickup_alpha` and `drop_bias` parameters that are not used. Status: fixed on 2026-06-01 by adding an inline code comment that these are legacy parameters and the current baseline uses `env.r_pick` plus nearest-pickup selection directly.
28. `step_towards()` always moves x-first then y and projects to the diamond if needed, which may distort the intended L1 route/reward near boundaries. Status: deferred by user decision on 2026-06-01. Preferred future option: replace x-first/y-first movement with L1 geodesic interpolation `from_pt + (max_dist / manhattan(from_pt, to_pt)) * (to_pt - from_pt)` when the target is farther than one step. Because the diamond is convex, this keeps movement inside `|x| + |y| <= R` when both endpoints are feasible and preserves the intended L1 step length.

## First-Round Verification Checklist

- Static checks should confirm `main()` is disabled, `HEUR_VOR` preserves original visible indices, `FOUR_ZONE` forces zone-route package selection, ride arrivals are sampled after action resolution, PPO dropout is removed, and `CoModalEnv` accepts an optional seed.
- Smoke checks should verify a small seeded environment is reproducible and both experiment notebooks can execute the core definitions without running full sweeps.

## 2026-09-19 Reconciliation of Issues 1-29

本轮依据当前四个主 notebook、README、progress_log 和既有研究决定重新核查。原问题大多有合理依据，但“原问题合理”不代表“当前仍未修复”，也不代表“必须改成论文原设定”。未修改实验或分析代码；未启动完整训练。9 月 5 日的环境/策略例子已重新执行，结论仍成立。

### 逐项结论

| 编号 | 是否合理、当前状态 | 现在是否需要修改 |
| --- | --- | --- |
| 1 | 原过时入口问题合理；main 已禁用 | 不需重修，仅更新旧状态 |
| 2 | 原 paper_axes 中未定义变量问题合理；已分离 CSV 读取 | 不需重修 |
| 3 | 原筛选后动作索引错配合理；现保留原索引 | 不需重修 |
| 4 | 原四区路线未执行问题合理；现已强制路线目标 | 原问题关闭；独立问题 A1-A4 也已于 2026-09-21 修复 |
| 5 | 原观测动作与新到达时序错配合理；已修复 | 不需重修，回归例子通过 |
| 6 | 原 PPO dropout 风险合理；已移除 | 不需重修 |
| 7 | 原环境局部 RNG 无种子问题合理；已修复 | 不需重修；不扩大为 GPU 全流程确定性保证 |
| 8 | 属于几何符号/模型约定差异；已决定保留 2R² | 保留决定；另明确 alpha 的物理半径换算（B4） |
| 9 | 原文档面积不一致合理；已统一 | 不需重修 |
| 10 | 与严格论文策略的差异确实存在；有意保留 pre-grace | 不改逻辑，论文描述需承认是修改版 |
| 11 | 原筛选参考点问题已按导师确认改成配送目标 | 原问题关闭；TTL 展示时限已确认，FOUR_ZONE 最终配送点筛选已于 2026-09-21 修复 |
| 12 | 参数不同是扩展研究范围，不是程序错误 | 按既有决定不限制到论文参数范围 |
| 13 | 原 eval 重复反缩放问题合理；现有显式缩放标志 | 不需重修默认流程 |
| 14 | 原训练/评估列数变化问题合理；已采用固定列 | 原问题关闭；重复运行混入同一日志是独立的记录管理问题 |
| 15 | 原 RT 标签与过滤不一致合理；当前目标图已过滤 RT=6 | 不需重修原问题；不要求所有其他图均固定 RT |
| 16 | 原 HVOR 最佳 alpha 与 DRL 平均 alpha 不可比问题合理；已统一双方择优 | 原问题关闭；参数选择与独立测试仍需分开（B8） |
| 17 | 问题合理；9 月 19 日确认分析绘图遗漏 | 2026-09-21 已补完两份 analysis notebook 的 PURE/PURE_OR 平均逻辑 |
| 18 | 5 个回合的统计精度担忧合理；它不是语法/实现错误 | 调试保留 5；最终按既有计划增加，并评估方差、独立测试和多训练种子 |
| 19 | 主目标采用总收益/总时间合理；当前已实现 | 不需重修；ep_rate 继续作为不同含义的诊断量 |
| 20 | 原绝对数据路径问题合理；现有可迁移解析器 | 原问题关闭；旧缓存位置和自定义小时曲线克隆是独立问题 |
| 21 | 原非平稳结果读错目录问题合理；现用 ns_results_path | 不需重修 |
| 22 | 显式小时周期特征是合理改进；当前已加入 | 不需重修；单凭此改动不能证明观测已是完整 Markov 状态（B2） |
| 23 | 固定起始时刻会限制结论适用范围，但若研究固定开工时间并非错误 | 保留暂缓；未来泛化到任意开工时间时再做稳健性分析 |
| 24 | 若声称 Manhattan 空间校准，城市总体时序不足；当前已明确只提取小时变化 | 保留抽象菱形与暂缓决定，不把时间校准说成空间复现 |
| 25 | 原 README 默认参数/入口过时合理；主要参数已更新 | 原问题关闭；仍有少量措辞/图轴单位清理，不等于默认参数仍错误 |
| 26 | 原包裹面积注释错误合理；已改为 2R² | 不需重修 |
| 27 | 参数未使用是接口清晰性问题；已注明 legacy，未删除参数 | 按既有决定保留；可选清理，不是算法错误 |
| 28 | 边界投影确实改变计费距离和时间；本轮新增数值例子 | 仍尊重暂缓；做严格物理解释或最终模型验证前应重新评估优先级 |
| 29 | 折扣累计回报与收益率目标确实不同，但不是自动证明算法无效或低完成率成因 | 保留暂缓；不改目标、奖励、截断规则或未完成惩罚 |

### 对暂缓问题 28 的新增证据

直接执行两个 notebook 当前的 `step_towards()` 与 `project_to_diamond()`。测试为 R=1、起点 (0,1)、终点 (1,0)、每步距离预算 0.095；两个端点均在合法区域内。结果一致：

- 最短 L1 距离：2。
- 对投影后位置逐步计算的实际累计 L1 距离：约 2。
- `step_towards()` 返回并用于载客收入的累计距离：约 2.845647。
- 实际实现需要 30 步，而按 0.095 的距离预算走最短路只需 22 步。

因此这不是单纯的绘图路径样式差异，也不是仅由 1e-6 级误差引起；先移动再投影会使位置变化、计费距离与固定速度设定不一致。该构造例子不代表随机实验中的发生频率或整体偏差大小，尚未测量其对各策略比较的净影响。记录证据不构成对用户暂缓决定的撤销。

### 2026-09-19 待办与后续处理状态

1. **旧清单中明确漏修：#17。** 已于 2026-09-21 补完，两份分析 notebook 的纯配送不再按 alpha 择优。
2. **旧清单中的后续实验要求：#18。** 最终评估精度和独立确认尚未落实；5/20/30 并非自动保证统计可靠性的阈值。
3. **#4、#11 相关的其他问题：** 四区接客后转区/连续接客/筛单窗口/路线成本（A1-A4）已于 2026-09-21 修复。B3 的 TTL 定义已于 2026-09-20 确认，保留展示时限，不要求实际 pickup deadline 检查。
4. **旧清单没有收录的新问题：** 时间分解 A5、散点分组 A6、最终评估时点 A8、自定义小时曲线克隆 A9、采样重置导致长回合覆盖缺口 B1、观测缺失 B2、alpha 换算与 TTL 饱和 B4-B5。确定实现错误和研究定义待明确的事项仍按该报告区分。
5. **保留/暂缓：#8、#10、#12、#23、#24、#28、#29。** 不把已接受的研究范围改动列成必须恢复论文原设定的 bug；#11 的已确认筛选中心也继续保留。

详细函数位置及原始复现说明见 CODE_REVIEW_2026-09-05.md；后续修复状态见下文。#28 有了更强证据但保持暂缓，#29 的因果推断仍没有证据支持。

## 2026-09-20 TTL Definition Decision

- 用户确认：TTL 是订单在系统中被接受前的展示有效期，保留当前规则。
- 接受订单后不再按该 TTL 使其过期，不构成 bug；B3 中“实际接客晚于展示到期时间”的例子保留为语义说明，不再作为待修复的接客截止时间问题。
- 现有从配送参考点计算 ETA 并与剩余 TTL 比较的额外可见性筛选也未修改；它不保证车辆的实际接客时限，B5 关于该筛选造成 alpha 饱和的观察仍成立。
- FOUR_ZONE 的筛单窗口 t0、是否在窗口内承诺具体订单，以及落客后的分区转换是独立问题，继续逐项讨论。本次只更新记录，没有修改 notebook。

## 2026-09-21 Issue 17 and FOUR_ZONE Fixes

- **#17 / A7：** 先修改两份分析 notebook：PURE/PURE_OR 按固定 lambda、gamma 下各次运行的 rate 平均，不按 alpha 选最大值；乘客策略仍先计算各 alpha 的运行均值再择优。
- **A1：** FOUR_ZONE 明确区分配送与接送阶段，接受订单时记录目的分区，落客后配送该区。每次清区最多对应一趟乘客任务；无订单则直接转向最近未清区，接送期间不筛选下一单。
- **A2：** 从整区预计完成前 t0 分钟开始，以区内路线最后一个配送点筛选；当场接受并保存具体订单，继续送完本区后接客。保留接受前 TTL、参考点 ETA 筛选、可见数量上限及现有半径映射。默认半径沿用 env.r_pick，显式 r_pick_alpha 参数现在会生效。
- **时间离散约定：** 在决策时间点首次满足“本区剩余时间 <= t0”时筛选，包含完成时刻；若本区耗时短于 t0，从开始该区服务时筛选。无订单时不在清区后额外等待。完成时间只预演现有确定性移动，不抽样未来订单；包含时间步取整和既有边界投影，未修改暂缓的 #28。
- **接受与开始接客：** 新增 pending_ride 状态和统一接受入口。接受时从展示列表移除并计数一次，等待配送期间不再过期或接受另一单，开始接客时不重复计数。reset 清空该状态，终止条件计入已接受但尚未开始的任务；PPO 观测维度保持原样。
- **A3：** FOUR_ZONE 路线距离改为乘 1000 后取整，与 PURE_OR 精度一致。
- **A4：** 两种 OR-Tools 路线都保留起点到第一件的成本，仅将人工返回起点的终止边成本设为零，优化固定起点、自由终点的开放路径。
- **验证：** tests/test_notebook_regressions.py 的 14 项测试全部通过，每项覆盖两种场景 notebook。包括分析聚合、筛单起点/中心、接受后 TTL、目的分区、三单上限、无需求/空区/重复点、截断/异常恢复、普通立即接客与索引兼容、真实 OR-Tools 成本及随机需求整段运行。随机非平稳测试直接使用非恒定小时曲线；未验证训练环境克隆曲线的 A9 问题。
- **运行环境：** OR-Tools 9.15.6755 安装于 tmp/fixes_2026-09-21/.venv 隔离测试环境；未安装或运行 PyTorch/GPU 训练。
- **兼容性与文件检查：** 两个版本各 200 步普通动作的观测、掩码、奖励和 info 与修改前完全一致；四份 notebook 所有代码单元解析通过，输出与元数据保持原样，两版本共享路线/FZ 函数一致，修改文件通过 diff 空白检查。
- 历史 notebook 输出、CSV/PNG 保持原样。完整扫描及收益变化尚未重算。A5/A6/A8/A9、B1/B2/B4/B5/B6/B7/B8 与已有暂缓事项不因本次修复而关闭。

## 2026-09-21 Remaining-Issues Recheck

用户要求复核修复后的剩余事项。这是 A5/A6/A8/A9 修改前的历史复核快照；该次只检查并更新记录，没有继续修改四份主 notebook。保留已关闭的 A1-A4、A7/#17 与已确认的 B3/TTL 定义；修正上方历史表格中 #4/#11 的过时待办文字。A5/A6/A8/A9 后续已按用户确认修复，见下一节；其余未关闭项继续保留。

| 编号 | 当前状态与本轮证据 | 后续处理性质 |
| --- | --- | --- |
| A5 | 复核时两个环境将落客的最后 0.5 分钟记为配送时间 | 后续已修复：按本步活动记账 |
| A6 | 复核时两份散点图把约 20 件标成 30、约 40/60 件标成 50 | 后续已修复：仅保留目标密度 |
| A8 | 复核时返回“更新 7 次的模型、第 5 次的评估指标” | 后续已修复：最终更新也评估、记录一次 |
| A9 | 复核时非平稳评估环境未复制自定义 hourly_multiplier | 后续已修复：复制实际小时曲线且不共享数组 |
| B1 | 复核时 collect_rollout 入口无条件 reset，长回合后段缺少训练样本 | 后续已修复：训练器保存采样状态，跨更新延续回合 |
| B2 | 原观测缺失已接受行程目标和可见订单剩余 TTL | 2026-09-21 已补齐这两类特征；保留最近 10 包裹及可见列表截断，因此仍是部分观测，性能影响待实验 |
| B4 | 现有面积约定和 r_pick 公式对应名义面积比例 alpha_code²/2 | 2026-09-21 按方案一保留半径；明确 alpha_paper=alpha_code/√2，并记录换算元数据 |
| B5 | 默认 ETA/TTL 筛选将距离限制在 0.95，较大 alpha 等价 | 后续已处理：保留筛选规则，扫描自动去重、显式多种子、分析先合并等价组 |
| B8 / #18 | 评估仍固定 5 个种子，同批结果参与选 alpha；未单独实现最终测试或多训练种子确认 | 调试保留 5；正式实验需增加回合、独立确认和方差评估 |
| B6 / B7 | Voronoi 分区不等面积，DRL 也不是论文全信息最优基准 | 论文解释边界；不自动要求修改已接受的策略变体 |

另有运行/文档事项：扫描汇总仅在末尾保存、无模型检查点；日志追加且无 run_id；eval 的 episodes 记录为一条聚合记录而非真实回合数；收敛图仍写 scaled；求解失败回退按距起点排序而注释写最近邻。完整训练环境/GPU 验证和修复后正式实验仍待完成。

继续暂缓 #23（开工时间）、#24（纽约空间映射）、#28（移动/计费距离投影）、#29（折扣目标与收益率）；保留既定几何、pre-grace、展示 TTL、参数范围与截断回合处理。

本轮复现输出：tmp/fixes_2026-09-21/remaining_results.json；脚本：check_remaining.py。A8 使用真实训练函数搭配轻量训练/评估替身验证控制流；其他行为例子运行真实环境或实际分析表达式，B1 为静态核查。没有执行 PPO 训练，不据此判断上述问题造成的收益/完成率变化。

## 2026-09-21 A5/A6/A8/A9 Fixes

- **A5（两份实验 notebook）：** `CoModalEnv.step()` 按实际执行的分支确定本步时间归属。前往接客和载客行驶均计入乘客服务时间，包括完成落客的最后一步；FOUR_ZONE 已接受订单但仍配送的等待阶段继续计配送时间。每步仍整体计入一个类别，未改移动、奖励或步长。
- **A6（两份分析 notebook）：** 散点图先用 `np.isclose`（绝对容差 1e-8、相对容差 0）保留 `GAMMA_PACK=0.50/0.83`，再赋予约 30/50 件标签；不再把其他密度就近归入这两组。图题明确为期望包裹规模；在 R=5.5 下对应 30.25/50.215，而非每回合实测件数。保留原 lambda、策略及 RT 混合选择。
- **A8（两份实验 notebook）：** 周期评估与最终评估共用 `evaluate_and_log()`。最后一次更新一定评估并记录；若它恰好是周期评估点则只执行一次。`eval_every=0` 仍评估最终模型；零次更新评估初始模型，并在提供日志路径时记为 update=0。返回模型与返回指标对应同一训练时点，日志列及缩放规则不变。
- **A9（非平稳实验 notebook）：** `_make_eval_env_from()` 传入训练环境 `hourly_multiplier.copy()`，保持相同的实际 24 小时需求曲线，避免数组共享导致相互修改。
- **验证：** 新增 tests/test_remaining_a_fixes.py 的 6 项行为测试，与既有 14 项一起运行，20 项全部通过。覆盖接客/载客/落客/配送等待、无效动作、散点筛选及实际绘图单元的数据传递、六种最终评估排程与真实 CSV 日志、默认/自定义曲线的 24 小时到达强度和独立数组。A8 使用真实训练函数和日志函数，训练更新/评估计算采用轻量替身；散点图用绘图记录器检查输入，未渲染历史图或运行 PPO。
- **兼容性：** 以本次修改前快照对照，两个环境各 1000 步的轨迹、观测、动作掩码、收益和终止逻辑一致；各有 12 次落客的最后一步从配送改记为乘客服务时间，总时间不变。四份 notebook 语法通过，保存的输出和元数据未变，共享 step、训练和路线函数保持一致。复核脚本：tmp/fixes_2026-09-21/verify_a_class.py。
- **状态：** A1-A9 现已全部关闭。B1/B2/B4/B5/B6/B7/B8、C 类运行/日志事项，以及 #23/#24/#28/#29 暂缓事项继续保留。未进行完整 PPO 训练、GPU 验证或参数扫描；历史 CSV/PNG 和 notebook 输出仍需后续重跑。

## 2026-09-21 B1/B5 Fixes

- **B1：** 训练器为每次训练创建独立 `RolloutState`，跨采样批次保存观测、掩码、回合累计收益/步数。仅真实结束回合时 reset；批次结束但回合未结束时仍 bootstrap，GAE 按批计算。完成回合的统计包含之前批次并仅记录一次，按环境实际 dt 换算时长；评估继续使用独立克隆。单独调用 collect_rollout 而不传 state 时仍采集新回合；不同训练环境不能共用 state。
- **B5 扫描：** 新增共享 experiment_support.py。每次按 R、V、DT 和 `max(1, round(TTL/DT))` 自动计算饱和阈值；同一其他参数组合内将等价 alpha 合并，选择请求列表中的最小值作为代表。默认每有效设置训练一次；`TRAIN_SEEDS` / `train_seeds` 支持显式多种子，各 alpha 使用同一组种子。当前默认平稳 420→240 组、非平稳 240→180 组（每种子）。未删除 ETA 条件或改变逐单剩余 TTL 筛选。
- **B5 记录：** 原始运行 CSV 和训练日志新增 train_seed、replicate、ALPHA_EFFECTIVE、ALPHA_SATURATION、ALPHA_MEMBERS、R/V/DT、实际 TTL 步数、需求曲线及训练设置。R_PICK_ALPHA 保留实际代表值，ALPHA_MEMBERS 保留请求的原始 alpha 列表。扫描额外写 `<csv_stem>_alpha_summary.csv`，按完整场景/等价组报告 rate_mean、rate_std、n_runs；单次运行标准差为空，n_runs 不是评估回合数，固定评估种子的基线重复不等于独立评估样本。
- **B5 分析：** 两份分析中的收益曲线和 HVOR/DRL 比值、两份实验 notebook 的最优 alpha 图，均先求等价组运行均值再择优。不同 lambda、gamma、TTL、RT、可见数、几何、需求曲线及训练设置等保持分组；宽范围概览在各固定场景分别择优后再平均。PURE/PURE_OR 保留问题 17 的运行均值规则。alpha 趋势图使用有效 alpha。
- **历史数据：** 新 CSV 依照逐行实际参数自动分组；旧 CSV 缺失 R/V/DT 时，由显式 LEGACY_GEOMETRY 提供历史假设并提示（当前为 5.5/.19/.5），不能从旧文件恢复不存在的元数据。旧 alpha 列不被覆盖，新元数据优先。此调整不使旧结果变成 B1 修复后的实验结果。日志列已增加，检测到旧表头时拒绝追加并要求使用新日志文件，避免破坏原文件。
- **验证：** 新增 tests/test_b1_b5.py 的 10 项测试，完整 30 项通过（28.0 秒）。两种真实环境均从 8192 步/4096 分钟继续到 11520 步/5760 分钟，并正确开始下一回合；检查跨批累计、真实终止/非终止 bootstrap、单步批次及环境隔离。两种场景均执行实际 ActorCritic、GAE、PPO 梯度更新和真实 evaluate_all（仅该测试的两种昂贵路线基线使用纯配送替身），确认中途评估不改变训练环境或 torch RNG。另覆盖参数/步长变化、逐单可见性、种子配对、分组均值/标准差、实际绘图输入、CSV 汇总和旧日志保护。
- **文件兼容：** 四份 notebook 语法与共享导入检查通过；可从项目根目录或 NonStationary 目录导入公共模块，原保存输出/元数据未变。CoModalEnv、FixedPackageEnv、FOUR_ZONE、环境克隆与 evaluate_all 的行为定义相对此轮修改前完全一致。脚本：tmp/fixes_2026-09-21/verify_b1_b5.py。
- **测试环境：** 使用已有 OR-Tools 9.15.6755。项目内长路径导致首次 CPU PyTorch 安装未完整完成；可用的 PyTorch 2.14.0+cpu 隔离依赖随后安装在 `%TEMP%/ll-b1-torch-20260921`，测试时将其置于模块搜索路径首位。沙箱无法读取该目录，测试经授权在沙箱外完成。这是局部 CPU 验证，不代表 RTX 5070/CUDA 或完整项目运行环境已经配置好。
- **当前剩余：** B2、B4、B6、B7、B8，以及 C 类未关闭运行/日志事项。B5 的多种子入口已具备，但独立最终测试集和正式多种子实验尚未执行，因此 B8/#18 继续保留。#23/#24/#28/#29 仍按既有决定暂缓；未运行完整参数扫描或更新历史 CSV/PNG。

## 2026-09-21 B2/B4 Fixes

- **B2：** 两种环境均在观测末尾加入 7 个已接受行程特征：提前承诺标志、接客目标相对坐标与 L1 距离、落客目标相对坐标与 L1 距离。覆盖普通立即接客和 FOUR_ZONE 先接受再完成本区配送；上车后清除接客特征，落客后清除目的地特征，reset 清空。保留已有接客/载客标志，以区分零距离目标与没有目标。
- **可见 TTL：** 每个可见订单由原 6 个特征扩为 7 个，加入剩余 TTL 步数 / 初始 TTL 步数。按请求对象身份匹配 TTL，排序后仍对应正确订单，重复坐标不会串单；可见列表上限及筛选不变。
- **维度与边界：** 默认平稳输入 67→79、非平稳 69→81；观测维度随最近包裹数与可见订单数自动计算。仍只保留最近 10 个包裹及有限可见列表，不能称为完整 Markov 状态。旧输入结构的模型需重新训练或另行设计权重迁移；本次没有测量收益/完成率改进。
- **B4 方案一：** 保留 `|x|+|y|<=R_code`、面积 `2R_code²` 和 `r_pick=alpha_code*R_code/√2`。相同物理菱形对应 `R_paper=√2*R_code`、`alpha_paper=alpha_code/√2`；未裁边界且未施加 TTL 时的名义面积占比为 `alpha_code²/2=alpha_paper²`。原扫描输入仍是代码尺度，不改变物理半径、ETA 条件或 B5 分组。
- **结果与日志：** 记录 `OBS_DIM`、`R_PAPER`、`R_PICK_ALPHA_PAPER`、`PICKUP_RADIUS`、`ALPHA_EFFECTIVE_PAPER`、`ALPHA_SATURATION_PAPER`。汇总提供有效 alpha 的论文尺度，不为包含多个名义 alpha 的等价组虚构单一名义值。不同已记录观测维度分别分组；旧日志表头保护继续生效。历史 CSV 可按原始几何派生换算，不修改文件或原名义 alpha；换算不表示恢复论文所有解析假设。
- **验证：** 新增 tests/test_b2_b4.py 的 8 项测试，完整 38 项全部通过（26.3 秒），覆盖接送生命周期、提前承诺、零距离、TTL 身份/排序/截断、维度/补零、保留的部分观测、物理半径与面积换算、扫描/汇总/日志。既有真实 CPU PPO 更新在新输入布局下通过，真实 OR-Tools 检查继续通过。
- **兼容性：** verify_b2_b4.py 对照本轮修改前快照，两种环境各 1000 步固定动作轨迹（各含 12 次落客）的运动、原有观测字段、mask、收益和终止结果完全一致。实际环境除观测方法外的所有既有方法保持相同；B5 种子、分组、阈值和数值汇总相同。四份 notebook 语法通过，保存输出/元数据未改，两个分析 notebook 仅补充尺度说明。
- **当前剩余：** B6/B7 的理论解释边界、B8/#18 的独立评估与正式多种子实验、C 类未关闭事项，以及 #23/#24/#28/#29 的原暂缓决定。B2 关键缺失已补，压缩观测限制明确保留；B4 参数定义按方案一关闭。未运行完整扫描、GPU 验证或重写历史结果。

## 2026-09-21 C1 Persistence and Progress

- **逐组保存：** 两份实验 notebook 的 run_param_sweep 每完成一组参数/训练种子，先提交该组六种策略的 result.json，再更新本次运行和原指定路径的结果 CSV/alpha 汇总。每个文件采用临时文件写完、刷新后原子替换；汇总导出失败可从已提交组重建，不必重复训练该组。
- **独立目录：** 每次新扫描自动生成 `<csv_stem>_runs/<UTC时间戳>_<唯一后缀>/`，启动即打印，完成后也在 df.attrs['run_directory'] 返回。保存 manifest.json、progress.json、运行结果，以及各 combo_XXXX 的独立日志和检查点。原 CSV 路径继续作为方便旧分析读取的导出；独立运行目录保留各次结果。新扫描不再追加公共 TRAIN_LOG_PATH。
- **检查点：** 默认每 5 次完整 PPO 更新保存 latest.pt，可通过 checkpoint_every 调整；初始化、周期评估完成和最终更新同样保存，最终完整模型另存 final.pt。包含模型、优化器、Python/NumPy/Torch CPU/CUDA RNG、环境及其独立 RNG、未完成回合的观测/累计值和已完成评估指标。用数据字段编码 RideReq，不把 notebook 中定义的类/函数对象写进检查点。
- **恢复：** 在原扫描调用加入 resume_dir=之前打印的目录，使用相同参数、种子、更新总数及代码版本；校验保存的扫描配置和组配置，跳过 result.json 已提交的组。未完成组从最近完整更新恢复；只回退该组独立日志中晚于检查点的记录，避免重复。中途打断优化器不保存半完成更新；最终评估失败保留已完成模型，只重试评估。零次更新和已全部完成后的再次恢复也有覆盖。旧代码运行无检查点，无法事后仅凭 CSV 续训。
- **进度：** 显示总组数/已完成组数、当前组 PPO 更新进度、阶段、耗时与估计剩余时间。可用 tqdm 时显示进度条，否则使用文本条；show_progress=False 只隐藏显示，不关闭进度文件。运行级/组级 progress.json 在阶段与更新边界更新，包含时间戳和最近保存更新；不是持续心跳，进度 ETA 不保证包含以后评估的精确耗时。
- **日志解释边界：** 独立目录已避免新扫描与其他运行混写，并支持安全日志回退。旧日志没有逐行 run_id，两个 analysis notebook 仍默认读取历史公共日志；画新训练曲线时需将其 log_path 指向目标组的 training_log.csv。跨运行合并、按 run_id/参数/种子分组、旧日志迁移尚未改造；episodes=1、scaled 图轴等其余 C 项也未改变。
- **验证：** 新增 tests/test_training_persistence.py 的 7 项测试，完整 45 项通过（35.7 秒）。真实 CPU PPO 在优化器已部分改变后中断，恢复模型、优化器、环境、所有已测 RNG 和日志与连续训练完全一致；另覆盖评估中断、零更新、配置拒绝、接单状态序列化、逐组恢复去重、原子写失败和进度降级。恢复测试的评估使用小型确定性替身，既有真实 evaluate_all/OR-Tools 回归仍通过。
- **兼容性：** verify_c1.py 对照本轮前快照，两份 notebook 除训练器和扫描函数外的全部函数/类定义保持一致，原输出/元数据保留，环境与 B1/B5/B4 共享逻辑未改。修改前后相同种子的实际 CPU PPO 权重与指标完全相同。
- **限制与剩余：** 新增共享 training_persistence.py，迁移时与 experiment_support.py、notebook 一起保留。检查点仅应读取自己生成的可信文件；尚未验证 CUDA 恢复、跨版本数值一致性、OneDrive 同步冲突或完整长跑，不支持两个进程同时写同一恢复目录。B6/B7/B8、其他 C 项及已有暂缓决定继续保留；没有覆盖历史结果。

## 2026-09-21 Paths, Run Identity and Archive Cleanup

- **四份 notebook 目录核对：** 新增 project_paths.py，统一平稳 Results/、非平稳 NonStationary/Results/。修复平稳版依赖 cwd 的 CSV/图片路径、默认扫描文件与 analysis 不同名、部分图片写到 notebook 旁边、旧公共日志入口及重复的非平稳路径查找器。根目录或 NonStationary cwd 均定位同一场景；外部启动可配置 LAST_MILE_PROJECT_ROOT。相对结果/检查点路径锚定本场景，绝对自定义路径仍保留。
- **数据目录：** Uber 自动查找会跳过没有匹配 parquet 的空目录，相对显式路径按项目根目录解析。当前兄弟目录 Uber_NYC 已确认含 12 个 2021 月度文件；小时缓存使用 NonStationary/Results/hourly_alpha_2021.csv。去掉未使用的顶层 PyArrow 导入，已有缓存可独立读取。本次未重读全年原始数据或拟合缓存。
- **run_id：** 每次新扫描生成独立编号；同次扫描的各参数/种子组通过 combo_id 区分。逐行日志、结果、alpha 汇总、manifest、组 run.json、检查点和进度均记录编号；续训保持原编号。独立 standalone 训练也会产生新编号，即使参数/种子重复。检查点格式升至 2，不自动迁移无编号的旧检查点。
- **analysis 新入口：** 两份收敛单元默认选本场景最新且有日志数据的运行，再选其最新有记录的组；打印实际 run_id、combo_id、train_seed 和绝对 log_path。可用 ANALYSIS_RUN_ID/ANALYSIS_COMBO_ID/ANALYSIS_RUN_DIR 显式指定；指定项找不到时不改选另一运行。读取后校验唯一运行/组/种子及更新记录，不再按 update 混合不相干训练。图轴单位按 REPORT_UNSCALED 显示。尚未正式训练时给出缺少新日志的明确提示，不悄悄读旧日志。
- **统计：** alpha 汇总将 run_id 纳入场景键；同一次扫描中的配对训练种子仍可组成重复样本，不同扫描不会自动合并。旧结果 CSV 没有 run_id 时不伪造身份，历史 CSV/PNG 与 notebook 保存输出保持原样。
- **旧日志与清理：** 清点找到的 5 份历史训练日志已经位于 archive/training_logs/{stationary,nonstationary}，活动结果目录没有公共训练日志，无需重复移动。回归通过后，将根 tests/、tmp/fixes_2026-09-21/（含隔离测试依赖和快照）、tmp/review_2026-09-05/ 归档至 archive/validation_2026-09-21/ 的 tests/、fixes/、review_2026-09-05/。保留检查依据，不删除项目运行需要的三个共享模块。归档清单记录移动路径及内容校验。
- **验证：** 新增 8 项测试，与既有 45 项合计 53 项；覆盖四 notebook 在两种 cwd 的初始化/文件定位、实际扫描默认输出路径、全部产物 run_id/恢复、独立训练身份、显式/默认日志选择、防重复、防跨运行混合、两个真实收敛单元绘图数据、需求目录搜索与缓存读取。既有 CPU PPO、OR-Tools 和精确中断恢复继续通过。verify_paths_runid.py 确认语法及原输出/元数据保留，环境、策略、评估、PPO 采样/更新定义不变。
- **剩余：** C 类运行身份与新日志连接已处理，收敛单位标签随单元更新修正；eval episodes 统计、PURE_OR 回退说明、核心代码复制等仍保留。B6/B7/B8、既有暂缓项、CUDA 与完整正式实验未在本次处理。
- **完成确认：** 完整 53 项在 29.8 秒内全部通过后才执行归档。移动 tests（13 文件）、审计材料（68 文件）、修复/依赖目录（8531 文件），文件数/字节数及源码证据 SHA-256 校验通过。归档后仍能发现 53 项测试，另重跑 8 项路径/身份测试全部通过；报告和清单保存在归档目录。根目录生成的字节码缓存与空 tmp 已清理，运行共享模块、历史 CSV/PNG 保留。
