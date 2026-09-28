# mask 正确率测试实验

## 目的与判定范围

验证程序生成的动作mask是否把应允许的动作标为1、应禁止的动作标为0。
暂停预算/优势强度调参，本实验沿用输入配置的物理参数、预算、gamma和学习率。

“应该允许”是指：在当前状态、当前有效的P1/P2/P3/P4上下文中，该动作满足单库条件，且能够延拓成完整五库动作路径。
下游行为mask还必须与已经选定的直接上游动作兼容。
因此，“本库局部可行但下游无路可走”的动作应标为0，并不属于误屏蔽。
生态等Layer 2长期约束不直接决定Layer 1 mask，不把生态不足直接作为mask错误。

松弛模式按当时有效的恢复边界判断，不按未经松弛的正常边界误报。
最终fallback只能选择规定的最小违反联合动作，不能把其他P1-safe动作误算为应该放行。

## 直接启动（项目根目录）

```bash
cd "/Users/hujingxin/Documents/Code_python/kelong/HARL"

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
/opt/anaconda3/envs/harl/bin/python -u -m examples.train_mask_accuracy \
  --load_config "harl/configs/experiments/cascade_budget_00001.json" \
  --updates 200 \
  --seed 1 \
  --print_every 100 \
  --exp_name "mask_accuracy_200"
```

这是实际HAPPO训练实验：200次训练更新，每轮完整2023—2025年1096天，共219200环境步。
每一步都检查全部201个候选动作以及4组201×201相邻兼容关系，不是每100步抽样检查。
`print_every=100`只控制报告频率；每轮训练结束还会再次打印，保留原reward/cost/lambda日志。
起始策略与最新/最终检查点按现有保存机制保存。新实验从头初始化，结果保存到新的时间戳目录。
运行时间取决于S3恢复频率和机器速度；包含审计开销，不能用短测试速度推断整个实验耗时。
如需更长实验，可以把`--updates 200`改为`--updates 500`（548000步）。

## 独立参考标签

`examples/mask_accuracy_oracle.py`不调用生产S201/S202/S203、ConstraintContext.resolve或S3求解器来生成标签。
它使用配置、原始Z-V锚点、动作映射范围、当前库容/历史下泄、外部及区间来水：

1. 独立重建线性离散动作下泄值。
2. 对每个候选动作正向计算V_next = V + (I-Q)dt。
3. 独立组合有效P1/P2/P3/P4边界，检查下一库容和泄量。
4. 对每个上游候选动作形成下游候选入流，计算全部候选动作对标签。
5. 用队列进行反向图可达性遍历，生成root与所有上游索引对应的条件mask。
6. 最终fallback用独立的路径标签算法比较累计P2/P3/P4违反量，完全并列时使用动作元组字典序。

数值比较使用现有实现约定：泄量容差1e-7 m³/s，fallback水位违反比较容差1e-8 m。
库容比较容差由泄量容差乘dt换算；不通过放宽物理参数消除差异。
独立标签与程序共用数据源，因此不能发现数据源本身是否符合真实工程；它检验的是给定数据与方法下的实现一致性。

## 实际检查位置

| 统计组 | 检查对象 |
|---|---|
| S201_root | WDD局部mask |
| S202_all_pairs | 下游四库在全部上游候选入流下的局部兼容矩阵 |
| S203_root | WDD完整可延拓mask |
| S203_all_conditions | 下游全部上游索引对应的条件mask |
| S3_behavior_root / conditions | 包含fallback强制动作处理后的行为mask |
| S4_actual_mask | 真实SequentialSampler保存的五个行为mask及实际选中动作 |

全部条件行包含训练中未选择的上游索引。对无法从root到达的上游索引，检查的是接口的“相邻兼容且后续可延拓”定义；不把这类行中的1解释成存在合法上游前缀。S4实际行为检查始终基于已选的合法前缀。

另检查恢复参数上下限、阶段顺序，以及进入更高级恢复前低级最大松弛是否仍无路。
本实验不是对S3最小松弛搜索精度的完整证明；已有S3专项测试继续承担精度测试。
不改变mask，不替换行为观测，不重新采样动作，不推进候选环境状态。

## 指标和判定

- TP：参考允许，程序也允许。
- TN：参考禁止，程序也禁止。
- FP（误放行）：参考禁止，程序却标1。
- FN（误屏蔽）：参考允许，程序却标0。
- accuracy = (TP+TN)/(TP+TN+FP+FN)。
- precision = TP/(TP+FP)。
- recall = TP/(TP+FN)。
- false_allow_rate = FP/(TN+FP)。
- false_block_rate = FN/(TP+FN)。

分母为0时写null/N/A，不伪造100%。
按模式、检查层、库分别统计；总指标合并多层比较，包含同一动作在不同层的检查，不能解释为独立样本的统计置信度。
大量禁止动作可能使accuracy看起来很高，因此必须同时查看FP、FN和recall。

合格目标：被检查样本中FP=0、FN=0，accuracy=100%，有正样本时precision/recall=100%。
一旦出现任一FP或FN，立即记录并停止，避免错误mask继续进入训练。

终端示例（格式示意，不是200轮已完成结果）：

```text
[MASK] steps=100 accuracy=100.00000000% FP=0 FN=0 modes={'normal': 100} status=running
```

## 输出位置

程序启动时打印`Mask reports: ...`，位于该次实验目录的`mask_accuracy/`：

- `summary.json`：累计结果、按库/模式/层统计、未覆盖模式、完成/失败/中断状态。
- `progress.csv`：定期累计快照，非逐动作明细，避免长实验产生海量文件。
- `first_mismatch.json`：第一处不一致的库、层、状态、参考参数和前30个错误索引。
- `first_mismatch_masks.npz`：完整程序mask和参考mask。
- `last_case.json`：发生异常/中断时的最近检查状态。

`status=running`或`interrupted`不表示完成；应检查最终是否为`completed`。
即使全部样本100%正确，也只能说明已访问状态通过。`uncovered_modes`非空意味着长训练没有覆盖对应分支，不能宣称全分支都在真实数据上验收。

## 已做的验证与额外测试命令

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
/opt/anaconda3/envs/harl/bin/python -m unittest examples.test_mask_accuracy -v
```

包含：

1. 手算水量平衡边界、P4生效、首日历史缺失与泄量数值容差。
2. 小动作图完整枚举与独立图遍历逐项一致。
3. normal/P4/P3/P2/fallback五种模式，以及P1硬不可行。
4. 人为注入一个误放行或误屏蔽，必须检出、报错并保存证据。
5. 相同种子的真实两轮短训练，对比审计开/关后的全部Actor与Reward/Constraint Critic参数，要求逐值完全一致。

这些短测试用于验证实验工具，不代替用户启动的200轮长实验。

另已完成真实数据下1096天的动作轨迹回放预检查（不更新策略）：S201/S202/S203及S3行为mask比较均为FP=0、FN=0。
实际覆盖normal 436天、P4松弛294天、P2松弛365天、fallback 1天；该轨迹未覆盖P3松弛，P3由上述专项构造测试验证。
预检查报告位于`results/mask_accuracy_replay_preflight/summary.json`。
该回放未重新调用Actor，S4真实采样mask已在训练中性测试中核验，200轮正式实验会在每个实际采样步继续核验S4。
