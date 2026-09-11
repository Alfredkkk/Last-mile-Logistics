# 当前代码与论文模型核查（2026-09-05）

## 结论与范围

优先处理四区策略的接客后状态转换、训练采样时域与评估时域的差异、散点图错误分组。四区路线距离精度、路线优化目标、时间分解统计和部分分析函数也存在确定问题。接客半径、TTL、观测信息和论文最优性结论的对应关系，则需要明确研究定义后再决定是否调整。

审查对象是当前四个主 notebook，不以 archive 或 .ipynb_checkpoints 为当前实现。已重新读取 progress_log.md、debug_log.md，保留其中截至 2026-08-07 的决定。论文主要依据本地 `Relevant Papers/4565248.pdf` 的第 7–12、16、24、26–28 页；核对正文并渲染检查了主要模型页面。

四个 notebook 的代码单元通过 Python 语法解析。对两个实验 notebook 提取原始环境和策略函数，执行了可复现的短例子；四区例子每区只有一个包裹，代替路线求解器的辅助函数只返回唯一访问顺序，实际环境和基线逻辑保持原样。最终评估时点的测试使用轻量替身验证训练函数控制流。没有运行 PyTorch 反向传播、OR-Tools 求解器或完整训练；目前对应依赖尚未配置。下面会区分实际复现、静态确认、数学推导和研究设计风险。

没有修改四个实验/分析 notebook，也没有覆盖历史 CSV、图或两份决策日志。源码定位均指原 notebook 的 JSON 文本行号；单元编号从 1 开始。

## A. 已确认的实现或分析问题

### A1. FOUR_ZONE 接客后未转入落客分区，且可能连续接客（高优先级，已复现）

位置：[experiment.ipynb](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/experiment.ipynb:1273>) 第 8 单元 `baseline_four_zone()`；[非平稳对应函数](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/NonStationary/experiment2.ipynb:1309>) 第 9 单元。

接受订单后 `current_zone` 保留为已经清空的旧分区，没有记录乘客目的地对应的下一分区。落客后可能再次执行 `pick_next_zone(env.pos)`，选出另一个分区；`screen_left` 也会在乘客行程期间继续运行，重新使 `have_candidate=True`，从而触发连续接客。

复现使用三个包裹：(0.05,0)、(-0.1,0)、(0,4)。车辆完成右区包裹，接受去上区 (-0.01,0.2) 的乘客；落客后却选择左区 (-0.1,0) 的包裹。如果再提供一个合格订单，实际事件序列为：送件 → 接客 → 落客 → 接客 → 落客 → 送件。两个 notebook 都复现。

论文第 10–12 页的 FZ 则在乘客目的分区继续配送，清空该区后才考虑下一次分区间乘客任务。这也是现有函数说明承诺的行为。现实现无法据此保证论文的最多 N−1 次接客结构。先修正状态转换，再评价 FOUR_ZONE 的性能；不能预先断言修复会提高还是降低收益率。

### A2. FOUR_ZONE 的筛单窗口并不是论文中的区域完成前 t0（高优先级，静态确认）

位置：[窗口逻辑](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/experiment.ipynb:1251>)，非平稳版相同。

只要当前路线上的下一个包裹一小步可达，代码就开启 `screen_steps` 窗口；没有要求它是本区最后一个包裹，也没有计算“距离整个区域完成还剩多少时间”。筛选参考点是当前路线目标，而非区域路线的最终配送点。`have_candidate` 只存布尔值，不保留筛选期间承诺的具体订单；旧窗口、旧候选标志也未在所有换区路径上清空。

论文第 10 页的逻辑是：本区预计完成前 t0 分钟开始筛选，以本区最终配送点为中心，接受合格请求，送完该区再去接客。代码在清空后重新选择当时仍可见的请求，所以 `screen_window_min=10` 不能解释为论文的 t0=10。

这项针对 FOUR_ZONE；不要求撤销你已决定保留的 HEUR/HEUR_VOR pre-grace 规则。

### A3. FOUR_ZONE 把小于 0.5 的距离舍入为 0（中优先级，已复现）

位置：[距离函数](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/experiment.ipynb:1079>)；[非平稳距离函数](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/NonStationary/experiment2.ipynb:1235>)。

`_l1_distance()` 使用 `int(round(distance))`，没有应用注释中提到的缩放。原函数对 0.49 返回 0，对 0.51 返回 1。大量短边会在求解目标中变成免费边，影响区内路线。而 PURE_OR 使用 1000 倍精度，两种路线基线并不一致。建议先统一距离精度。

### A4. 路线优化按闭环求解，环境按开放路径执行（中优先级，源码和官方文档确认）

位置：[PURE_OR](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/experiment.ipynb:972>)、[区内路线](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/experiment.ipynb:1110>)；两个实验 notebook 共有。

两处均使用 `RoutingIndexManager(n_nodes, 1, 0)`，并对回到起点的边计入正常距离。这个单一 depot 模型包含回程；区内路线注释所说的“默认允许不回 depot”不成立。实际执行却在最后一个包裹送完时结束，不执行回程。因此优化的路线成本与最终统计的成本不同。

枚举例子：起点 (0,0)，三个包裹 (1,0)、(-1,0)、(0,2)。不同访问顺序的闭环成本同为 8，实际开放路径成本却有 6 和 7。它说明闭环最优顺序也可能不是实际执行目标下的最佳顺序，不代表真实 OR-Tools 一定返回哪一种顺序。

若采用当前“最后一件完成即结束”的任务定义，应优化开放路径；无需为了这项修复强制车辆回仓。API 依据：[Google OR-Tools 起终点说明](https://developers.google.com/optimization/routing/routing_tasks#setting-start-and-end-locations-for-routes)。

### A5. 完成乘客行程的最后一步被计为配送时间（中优先级，已复现）

位置：[时间归属](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/experiment.ipynb:486>)；[非平稳对应位置](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/NonStationary/experiment2.ipynb:641>)。

落客时先设置 `with_passenger=False`，随后按更新后的状态判断本步属于载客还是配送。实测一段在本步结束的乘客行程产生载客收入，但该步 0.5 分钟全部增加到 `time_delivery_min`，`time_rides_min` 增量为 0。每次正常完成的乘客行程都会发生这种错分。

这主要影响时间结构解释，不会单独改变总时间或本步收益。应按照本步实际执行的活动记账。

### A6. n=30/50 的散点图混入其他包裹规模（高优先级，已复现）

位置：[analysis.ipynb 第 8 单元](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/analysis.ipynb:815>)；[analysis2.ipynb 第 8 单元](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/NonStationary/analysis2.ipynb:835>)。

代码对全部数据应用“向最近的 30 或 50 取整”，不是筛选 n≈30、50。实际映射如下：

| GAMMA_PACK | 期望包裹数 60.5γ | 散点图标签 |
| --- | ---: | ---: |
| 0.33 | 19.965 | 30 |
| 0.50 | 30.25 | 30 |
| 0.67 | 40.535 | 50 |
| 0.83 | 50.215 | 50 |
| 1.00 | 60.5 | 50 |

这会直接混淆“相同包裹规模下的收益—终止时间关系”。应先筛选 γ=0.50、0.83，再给近似 n 标签。另需明确该散点图现在混合 RT=5.5/6.0，并使用 PURE；前面的论文风格收益图使用 PURE_OR。混合/基线选择不一定错误，但应在图注交代。

### A7. 已决定的“纯配送不优化 alpha”没有覆盖全部分析函数（中优先级，已复现）

位置：[analysis.ipynb 的 best_alpha_series](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/analysis.ipynb:134>)、[best_alpha_by_lambda_gamma](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/analysis.ipynb:301>)；非平稳分析版相同。

旧问题 17 只在两个实验 notebook 的 `plot_rate_vs_lambda_optimal_alpha_by_gamma()` 中修好。两个分析 notebook 仍把 PURE_OR 传入按 alpha 取最大值的函数。

对原函数输入同一 PURE_OR 设置下 rate=1、3 的两条 alpha 记录，得到 3，而已有决定要求平均值 2。当当前各 alpha 的纯配送结果完全相同时，数值影响为零；存在样本或求解差异时，会择优选中噪声。这是旧修复遗漏，不是重议既有决定。

### A8. 某些更新次数会返回旧模型的评估结果（较低优先级，控制流测试复现）

位置：[train_policy_brief 结尾](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/experiment.ipynb:1763>)；非平稳版相同。

训练结束只在 `last_metrics is None` 时评估。测试 updates=7、eval_every=5，返回的是更新 7 次的模型和第 5 次的指标。当前 40/50/200 次、每 5 次评估的扫描不会触发，但修改训练长度可能静默混淆模型与结果。建议最后一次更新总是评估或显式返回评估时点。

### A9. 自定义非平稳需求曲线没有传递到评估环境（条件触发，已复现）

位置：NonStationary/experiment2.ipynb 第 11 单元 `_make_eval_env_from()`。

环境构造器支持 `hourly_multiplier`，但克隆评估环境时没有传入 `env.hourly_multiplier`，会回退到默认小时曲线。用平均值为 1 的自定义曲线构造训练环境后，原克隆函数没有保留该曲线。若一直使用默认 Uber 曲线不会触发；以后测试其他需求形状时，会出现训练和评估使用不同需求曲线的问题。应在克隆时显式复制该参数。

## B. 训练覆盖、模型信息与理论对应

### B1. 训练轨迹最多到 4096 分钟，评估却到 5760 分钟（高优先级，静态确认，影响待实验）

位置：[collect_rollout](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/experiment.ipynb:1381>)；[非平稳对应函数](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/NonStationary/experiment2.ipynb:1537>)。

每次调用函数时无条件 `env.reset()`；采满 PPO_STEPS 后，未结束回合不会保存下来供下一次更新继续。默认 8192×0.5=4096 分钟，评估上限则为 5760 分钟。于是最后 1664 分钟（约 27.7 小时）的状态没有直接的训练转移样本；末步只会计算一次 bootstrap value。完成较早的回合会正常重置，因而这里只针对长回合。

不能把“rollout 截断并 bootstrap”本身说成 PPO 数学错误；例如 [Spinning Up 实现](https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/ppo/ppo.html) 也包含采样边界截断。具体风险在于本项目 time_frac 入模且评估时域显著更长，训练不断重新回到起始时间。其对完成率、末期决策的实际影响尚未测试。

建议优先让采样器跨更新延续未终止回合，或者明确采用并验证不同的训练时域设计。这项与已暂缓的“折扣回报 vs 收益率目标”是两个独立问题，无需改奖励即可处理。

### B2. 观测缺少已接受乘客目的地等信息，不能当作完整 Markov 状态（中优先级，已复现）

位置：[环境观测](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/experiment.ipynb:362>)；非平稳版相同。

接受订单后，该订单从可见 buffer 移除。观测只保留接客/载客标志，没有明确编码 `to_pickup`、`drop_target`；请求剩余 TTL 和超过最近 10 件的包裹位置也没有完整输入。

实际构造相同位置、时间、包裹、可见订单的两个载客状态，只改变乘客目的地：输出观测完全相同，但下一步奖励不同，一个状态落客、另一个继续载客。虽然忙碌期间动作被忽略，价值函数仍需预测接下来多久恢复自由决策，因此信息缺失有实际意义。

论文第 9 页的一般动态规划状态包含全部未送包裹及相关乘客起终点。当前策略适合描述为使用局部/压缩观测的 PPO，不宜描述为完整状态动态规划的数值解。可以选择补充关键行程信息、引入记忆，或保留部分可观测设计并说明限制；尚无证据说明这是低完成率的主因。

### B3. “从配送目标筛可见订单”不等于“车辆能在 TTL 前接到乘客”（定义待确认，已复现）

位置：[ETA/TTL 筛选](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/experiment.ipynb:314>)；非平稳版相同。

复现：车辆 (0,0)、配送目标与 pickup 都在 (4,0)，订单剩余 TTL=0.5 分钟。代码从目标点计算 ETA=0，订单可见且可接受；车辆实际 21.5 分钟后才接到乘客。接受后不再跟踪该订单 TTL。

若 TTL 只是“未被接受前的展示时限”，可以保留此语义，但不能宣称它保证实际乘客等待时限。若 TTL 是 pickup deadline，则缺少车辆到接客点的可行性检查。对论文那种“先完成配送再接客”的策略，还需包括剩余配送时间。

这不建议撤销已确认的问题 11：可见范围仍可围绕配送目标。待明确的是可见范围与实际接客截止时间是否应该使用不同检查。

### B4. 保留 2R² 几何后，alpha 对应论文的比例还差一个 √2（理论差异，数学确认）

位置：[r_pick 定义](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/experiment.ipynb:220>)，两个实验 notebook 相同。

不重议旧问题 8 的几何决定。记代码 L1 半径为 Rc，论文边长为 Rp，则同一物理菱形对应 Rp=√2Rc。论文 r=alpha×Rp/√2=alpha×Rc；现代码却用 r=alpha×Rc/√2。

因此在不受边界裁切、也不施加 TTL 等附加过滤时，代码 pickup 区域面积占比为 2r²/(2Rc²)=alpha²/2，而论文第 10、12、16 页使用 alpha²。要映射到论文中的同一筛选比例，当前代码参数对应 alpha_paper=alpha_code/√2。

可选择保留代码并明确重参数化，或只调整 alpha 到物理半径的映射。不能在保持当前 r_pick 的同时原样套用论文 alpha²lambda 公式。论文第 10 页脚注 3 还允许边界外 pickup 以简化边界情形，代码只在域内生成请求；即使解决比例换算，边界处理仍是另一项差异。

### B5. 默认 TTL 让 alpha≈0.244 以上的半径扫描全部饱和（研究解释问题，推导和运行均确认）

代码同时要求 pickup 距离≤r_pick 且距离≤V×剩余 TTL。默认 V=0.19、最大 TTL=5，故任何可见订单的距离都≤0.95。对应 alpha 阈值 0.95√2/5.5≈0.24427。

所以当前平稳扫描的 alpha=0.25、0.30、0.35、0.40 在环境可见集合方面完全等价；非平稳扫描中的 0.30 和 0.40 同样如此。固定种子、相同行动，两个实际环境在 alpha=.25 与 .4 下连续 200 步的观测、mask、奖励完全相同。

run_param_sweep 对每个 combo 改变训练种子，所以等价环境的 DRL 结果仍可能不同。这时“最优 alpha”可能部分是在多个训练随机结果中择优，而不是更大半径带来真实改善。无需重新限制到论文参数范围，但应识别等价参数并单独进行多种子训练统计。

### B6. Voronoi Switching 是论文思路的变体，解析概率不能原样继承（理论差异）

论文第 9、16 页为解析推导采用等面积分区及各区相同包裹量，因而剩余 N−i 个区域的目的地概率为 (N−i)/N。当前 HEUR_VOR 以随机包裹点建立 Voronoi 区域，面积通常不同，并且哪些包裹被先送达与空间形状有关。

因此每个状态下，落客仍在未服务区域的概率应由未服务区域面积决定，不能自动等于未送包裹数量占比。它可以是合理的模拟策略，但不等价于论文等面积 N=n 的全部解析假设。加上已经接受的 pre-grace 变体，应明确使用“modified/Voronoi switching”的名称；不应直接沿用论文 N−1 接客限制和收益公式而不再验证。

### B7. HEUR_VOR/DRL 是经验策略比值，不是论文的最优性差距（理论解释边界）

论文第 26–27 页的 full-information benchmark 知道全部乘客请求，并放松等待与到达顺序限制。DRL 只能看到当时的局部观测，没有全信息，也没有全局最优保证；同时它的策略约束与 zoning 不同。

所以当前 `R_switching/R_DRL` 图可以比较两种实测策略，但不能解释为论文的 `R*/R⋄` 或“距理论最优百分之多少”。同样，论文关于近最优 zoning 的结论不能直接套到当前 DRL 比较。代码未自动宣称这一点，此处是科研表述应守的边界。

### B8. 固定五个评估种子，又用同批结果选择最优 alpha，缺少独立确认（统计风险）

位置：[evaluate_all 的 seeds](<C:/Users/93541/OneDrive - Georgia Institute of Technology/GeorgiaTech/Research/Last-mile Logistics/experiment.ipynb:1517>)；非平稳版相同。

每次都使用 42–46。不同策略共享种子适合配对比较，固定评估集也能减少收敛图噪声；这两点本身不是错误。但是同五个场景被反复用于评估、选 alpha 和报告最终效果，且每组只训练一个随机种子，不能据此估计泛化性能或训练方差。

旧问题 18 已决定调试先用 5、最终提高到 20/30，应继续遵循。新增建议是把选参数与最终确认的数据分开，并对关键组合重复训练；不要求每次训练评估都更换种子。

## C. 较低优先级的运行和文档事项

- 扫描汇总 CSV 只在全部组合结束时写入，模型未保存；中断后不能从已训练权重继续。适合正式长跑前加入逐组保存和独立 run 标识。它不改变科研模型。
- 训练日志采用 append，重新运行同一批会继续追加，没有 run_id；分析又按 update 汇总，可能把不同训练运行混在一起。需要独立文件名或显式 run_id。
- eval 日志把已经聚合的五个回合包装成一条 ep_stats，写出的 `episodes=1` 是记录条数，不是真实评估回合数；不影响当前 rate 计算，但不应把它作为评估样本量。
- 收敛图轴仍写 `Revenue rate (scaled)`，而正常 REPORT_UNSCALED=True 流程写入的是已反缩放的数值。
- PURE_OR 的失败回退是按距起点排序，不是逐步最近邻；固定两秒局部搜索也不保证最优，应称 OR-Tools 路线启发式。
- 当前两个 notebook 的核心逻辑基本复制，以上共享问题都需同时处理；单边修改容易再次分叉。

## D. 已有决定：保留或暂缓，不计为本轮必须修复

| 日志事项 | 当前解释 |
| --- | --- |
| #8–9 菱形面积 2R² | 保留，不要求改回论文符号；B4 另行说明 alpha 的换算 |
| #10 pre-grace switching | 保留；A1–A2 针对独立的 FOUR_ZONE 实现问题 |
| #11 可见集合以配送目标为参考 | 保留；B3 只指出真实 pickup 截止时间并未保证 |
| #12 扩展参数范围 | 保留，无需局限原论文数值图 |
| #18 调试评估 5 回合 | 保留，最终按已定计划增加；B8 增补独立确认建议 |
| #23 非平稳固定起始时刻 | 暂缓 |
| #24 不进行 NYC 空间映射 | 暂缓，继续仅拟合小时需求形状 |
| #28 移动路径及边界投影 | 暂缓；A5 是独立的时间归属错误 |
| #29 折扣回报与收益率目标差异 | 暂缓；不改 PPO 目标或奖励，不认定其为低完成率原因 |
| 截断回合继续报告 rate、无未完成惩罚 | 保留，同时报告 finish_rate |
| 修复前 CSV/PNG 暂留 | 已明确是历史输出，不作为新代码错误重复提出 |

## E. 复核后没有重新报错的已修复项目

- `main()` 已禁用，实际入口是参数扫描。
- `HEUR_VOR` 过滤后保留原始 visible 索引。
- FOUR_ZONE 已强制执行区内路线，旧“路线完全被忽略”问题已修好；A1–A4 是剩余的不同问题。
- 新到达请求放在动作执行之后；两个环境均通过订单索引回归例子。
- PPO 网络 dropout 已移除。
- 两个环境通过固定种子的轨迹重复性检查。
- 主收益率聚合现为 sum(reward)/sum(time)，并保留 ep_rate。
- eval 重复反缩放修复、固定日志列、RT=6 过滤、HVOR/DRL 双方采用 best-alpha 等改动仍在。
- 非平稳路径解析和周期小时观测已在当前代码中。

建议顺序：先处理 A1/A2/A3/A4/A6/A7 等确定问题并保持研究模型不变；决定 B1 的训练覆盖方案；再明确 B3/B4/B5 的参数含义和 B2 的观测范围；之后才进行完整实验及论文级对比。
