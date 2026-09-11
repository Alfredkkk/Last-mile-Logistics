# 项目交接记录（2026-09-04，Windows 台式机）

## 阅读范围与限制

递归盘点研究目录，排除 .git 内部文件后共 139 个文件：11 个 notebook、2 个 Python 脚本、3 个 Markdown、10 个 CSV、3 篇 PDF、110 张 PNG。解析 notebook 源码与保存的输出、CSV 全部记录和 PDF 文本；PNG 校验后按内容去重为 105 张，并通过缩略总览检查。此轮是项目接手梳理，不是逐行代码审计或实验复现。

用户启动 OneDrive 后，readme.md、debug_log.md、progress_log.md 均可直接读取。重新读取确认：progress_log.md 工作副本比 Git HEAD 多出 2026-08-07 Re-Review Decisions，内容与 debug_log.md 一致，确认旧结果未重跑、不添加未完成包裹惩罚、暂缓 PPO 目标差异研究。最近可见提交为 603137f（2026-06-09，Modify visibility logic）。后续以已读工作副本中的较新决定为准。

## 研究问题与模型

研究单辆车如何协调包裹配送和即时乘客订单，用 PPO 策略对比启发式及路径优化基线。服务域为 |x|+|y|<=R 的菱形，距离为 L1，面积采用 2R²；R=5.5 时面积 60.5。包裹初始数量来自空间泊松模型，代码将数量下限设为 1；实际数量随机，图中的 n≈20/30/40/50/60 是期望数量标签。

每步 DT=0.5 分钟，速度 V=0.19，回合最长 5760 分钟。包裹收入 RP=2/件，乘客收入 RT 按载客距离计算；内部奖励缩放为 1/8。接客和载客期间自动继续行程。包裹全部完成且没有进行中的乘客任务，或者达到时间上限时终止。

平稳场景订单强度恒定；非平稳场景用 Uber NYC 2021 数据拟合 24 小时订单占比，以 lambda_eff(t)=lambda×24×alpha(hour) 调节到达率。这里的小时占比 alpha 与接客半径参数 R_PICK_ALPHA 不是同一个量。纽约数据用于时间变化，不用于把实际城市路网映射进菱形。

## 当前入口与算法

- experiment.ipynb：平稳环境、策略、训练、评估、参数扫描和部分绘图。
- analysis.ipynb：平稳结果比较、收益率、终止时间、收敛及比值图；主要是数值分析，不是完整理论证明。
- NonStationary/experiment2.ipynb：非平稳需求拟合和训练；观测增加周期性小时 sin/cos 特征。
- NonStationary/analysis2.ipynb：非平稳分析。
- Results/、NonStationary/Results/：历史实验汇总及图。
- archive/：旧代码、旧参数实验、旧训练日志；.ipynb_checkpoints 为历史检查点，不作为当前入口。

六种策略：DRL（PPO）、HEUR（送件附近时机切换接客）、HEUR_VOR（增加 Voronoi 落客限制）、FOUR_ZONE（四区路线策略）、PURE（最近邻纯配送）、PURE_OR（OR-Tools 路线优化纯配送，不能直接称为保证最优解）。

PPO 为两层 256 单元 MLP，LayerNorm/ReLU，共享主干及策略/价值头；无效动作掩码，训练采样、评估贪心。默认 PPO_STEPS=8192，PPO_EPOCHS=4，minibatch=256。main() 已禁用；实际入口是 run_param_sweep()。

当前平稳扫描为 420 组×50 次 PPO 更新；非平稳代码为 240 组×40 次更新。每组独立初始化和训练策略。默认评估 5 回合；既有记录建议最终扫描使用 20 回合，小规模确认使用 30 回合。

## 已有数据的状态

| 文件 | 参数组合 | 数据行 |
| --- | ---: | ---: |
| Results/param_sweep_results.csv | 144 | 432 |
| Results/param_sweep_results_2.csv | 420 | 2520 |
| Results/param_sweep_results_2_full_param.csv | 720 | 4320 |
| NonStationary/Results/param_sweep_results_2.csv | 200 | 1200 |

主结果 CSV 均不含当前代码新增的 ep_rate。debug_log.md 明确允许修复前 CSV/PNG 暂留，等待重跑。当前两个 Results 目录均未发现新的 training_log.csv，收敛绘图依赖尚未满足。旧日志存在列宽不一致、缩放历史问题，不能因为文件名带 clean 就当作最新可靠结果。

非平稳历史 CSV 不含 lambda=30，而当前代码已包含；对应 rev_rate_vs_n_lambda30.png 是空图。

以下仅为历史主 CSV 中各算法跨参数组合的算术均值，用来识别现象，不是最新 time-weighted rate 的重新评估，也不是显著性结论：

| 场景与算法 | 已存 rate 均值 | finish_rate 均值 |
| --- | ---: | ---: |
| 平稳 DRL | 0.8612 | 42.95% |
| 平稳 HEUR | 0.6681 | 86.48% |
| 平稳 HEUR_VOR | 0.6520 | 100% |
| 平稳 PURE_OR | 0.2885 | 100% |
| 非平稳 DRL | 0.8016 | 60.50% |
| 非平稳 HEUR_VOR | 0.5560 | 100% |

finish_rate 是回合中完成全部包裹的比例，不是包裹送达件数比例。DRL 高收益伴随低回合完成率，是核心待解释现象；不能仅凭以上均值宣称 DRL 全面更优。

## 继承的研究决定

1. 保留菱形面积 2R² 的项目约定，不改回论文的 R 定义。
2. 保留有意设计的 pre-grace switching heuristic。
3. 可见乘客请求从当前配送目标位置筛选距离和 TTL，而不是从车辆位置筛选；观测中相对坐标仍相对车辆。
4. 主指标 rate/avg_rate=sum(reward)/sum(terminal_time)；ep_rate 为逐回合收益率均值，仅作诊断。
5. 已修复可见订单索引、FOUR_ZONE 路线执行、动作与订单到达时序、PPO dropout、随机种子、日志 schema 和重复反缩放等问题。
6. 保留时间截断回合的收益率，不增加未完成包裹惩罚，通过 finish_rate 暴露完成情况。
7. 2026-08-07 明确暂缓研究 PPO 折扣累计回报与最终收益率目标的不一致；尚未确立它是完成率低的原因。后续工作不要擅自改奖励或优化目标。
8. 起始时刻固定、纽约空间过滤、移动边界投影等问题暂缓。

## 论文资料

- 4565248.pdf：Cao、Liu，Coordinate Package Delivery and On-Demand Rides: A Zoning Policy and Analysis，60 页。项目主要模型及分区/切换策略来源。
- 1-s2.0-S0377221714002173-main.pdf：Li 等，The Share-a-Ride Problem: People and parcels sharing taxis，2014，10 页。SARP/FIP 与 MILP 路线建模背景。
- 1-s2.0-S0965856422003317-main.pdf：Fehn 等，Integrating parcel deliveries into a ride-pooling service—An agent-based simulation study，2023，26 页。乘客与包裹融合的仿真和车队应用背景。

## Windows 迁移核查

- 系统已识别 NVIDIA GeForce RTX 5070，显存 12227 MiB，驱动报告 610.88。
- 当前代码已有 CUDA 自动选择分支；但此轮没有完成 PyTorch CUDA 运算验证。
- 当前命令环境未找到 python/py/conda/uv；这不等于已证明整台电脑未安装 Python。Codex 自带 Python 可做资料检查，但缺少 torch、ortools、pyarrow、matplotlib、seaborn、jupyter、ipykernel，不能作为已经配置好的项目训练环境。
- 未发现 requirements.txt、environment.yml、pyproject.toml 或项目虚拟环境。
- OneDrive 启动后重新检查：相邻 ../Uber_NYC 的 2021 年 12 个月 parquet 均成功打开并读取首尾字节，全部具有 PAR1 格式标记；此前可读性障碍已解除。本次未逐行解析全部订单，因此不等于完整数据质量验证。旧 hourly_alpha_2021.csv 已完整读取，24 个小时齐全，占比之和为 1.0；最低为 4 时，最高为 18 时。
- 当前非平稳代码的缓存目标是 NonStationary/Results/hourly_alpha_2021.csv，此处尚无缓存；已读旧缓存位于外部 Uber_NYC，不能假定现有代码会自动读取它。配置非平稳运行时需安排缓存复用或从现已可读的 parquet 重新拟合。
- 未找到模型权重文件；两个主训练 notebook 未实现 torch.save/torch.load。每轮扫描结束才写汇总 CSV，目前不能按已保存模型恢复训练。
- 当前仿真和 OR-Tools 基线运行于 CPU，rollout 按单环境逐步执行，网络较小。仅凭 5070 无法预测加速幅度，需先测量。

建议后续执行顺序：确认/创建独立项目 Python 环境并记录依赖；验证 GPU 张量运算及单次 PPO 更新；小参数训练与评估；补充逐组落盘和模型检查点后运行完整扫描；用新数据生成最终图表。此轮没有安装软件、修改实验代码或启动训练。
