# Step 7 完整训练链路分支验收

本轮依据《论文与代码统一上下文_严谨修订版.md》第 26 节，验收各安全恢复分支及长期约束整批未激活场景。

## 1. 本轮实际修改

完善已有 `examples/test_cascade_branch_integration.py`，新增以下检查：

- 每个场景在 Critic 更新前保存 A_L，逐个 Actor 核对收到的优势不变。
- 每个 Actor 更新后，在保存的 observation/mask/action 下计算最终 new/old ratio，使用保存的行为 log probability 作分母，并核对后续 Actor 的 factor 是前序 ratio 的累乘。
- 正常场景要求实际出现非单位 ratio，避免全 1 的假通过；纯最终兜底场景验证单一可行动作的 ratio 为 1。
- 未激活场景构造非零 ξ，验证它被保留、训练 cost 为零、初始非零 λ 不变。
- 增加实际执行动作的 P3 日下泄变化检查及 P2 上/下最大松弛幅度检查。

本轮覆盖场景未发现新的运行逻辑错误，因此没有新增生产算法修复。没有改动 AGENTS.md、生产 YAML 数值、数据文件或已运行的训练进程。

## 2. 测试场景与边界

每个可训练场景用独立 Runner，真实执行 4 个连续决策步，并完成一次 compute/train/after_update。

使用真实的五个 Actor、Reward Critic、全部 Constraint Critics、安全求解器、Z-V 曲线、201 个动作映射、物理转移、Buffer 和优化器。未伪造 SafetyRecoveryResult、动作或环境 transition。

为了确定触发罕见分支，测试在内存副本中调整输入或约束：

| 场景 | 测试夹具 | 本轮实际模式序列 |
|---|---|---|
| 正常 | 项目默认初始场景 | normal × 4 |
| P4 松弛 | 从现有动作网格和 Z-V 推导测试来水及日水位变化限额 | p4_relaxed × 4 |
| P3 松弛 | 缩小内存中的 WDD 日水位变化限额 | p3_relaxed → fallback → p3_relaxed → fallback |
| P2 松弛 | 根据 P3 最大松弛下候选动作推导测试用 P2 上限 | p2_relaxed → normal → normal → normal |
| 最终兜底 | 将测试用 WDD 日水位变化限额设为极小值 | min_violation_fallback × 4 |
| 整批未激活 | 一月执行，仅在汛期激活；测试初始 λ=0.5；第一项测试要求设为 1e6 使 ξ 非零 | normal × 4，χ/v 全零，λ 不变 |
| P1 硬不可行 | 测试来水超过 P1 可接纳水量与最大下泄之和 | collect 明确报 No P1-safe，不进入后续训练 |

所有夹具数值仅用于测试，不是新的工程参数依据，也不写回配置或数据。
P3/P2 场景只要求首日触发指定分支，后续按真实状态继续推演，允许自然切换模式。

## 3. 贯通检查

```text
reset / warmup
→ prepare_step
→ 真实 WDD→THR 条件采样
→ env.step
→ insert
→ compute
→ Reward Critic / Constraint Critics
→ 随机顺序 HAPPO Actors
→ lambda
→ after_update
```

逐层检查：

1. prepare_step 与 collect 不推进日期、库容、历史下泄。
2. 五个 Actor 实际调用顺序符合水力方向。
3. 联合动作属于完整条件路径；最终兜底动作等于强制完整路径。
4. 不合法动作被拒绝，拒绝后物理状态不变。
5. Q_exec 等于 Q_target；下游入流为当前直接上游实际下泄加区间入流。
6. 水量平衡、P1 水位、有效 P2/P3/P4 及松弛幅度符合当前结果。
7. 每个 Actor 的 observation、mask、action、old_log_prob 与实际采样同索引。
8. Reward/Constraint Critics 使用 prepare_step 中同一个 x_t。
9. ξ、χ 同索引存储，cost 重建正确，returns 有限。
10. 所有 Critics 先于 Actors 更新，λ 最后更新；Actor 顺序与水力顺序分离。
11. 本轮 A_L 和 lambda_snapshot 固定，最终概率比正确累乘。
12. 未激活约束不更新 λ；after_update 清理本批 ξ/χ/cost。

测试只固定五 Agent 的更新排列，保留真实 minibatch shuffle，避免替换整个 torch.randperm 后误将样本索引变成 Agent 索引。

## 4. 执行命令

项目根目录执行：

```bash
conda activate harl
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONDONTWRITEBYTECODE=1
python -m examples.test_cascade_reservoir
python -m examples.test_cascade_reservoir_step3
python -m unittest examples.test_cascade_branch_integration -v -f
```

其中 `-f` 表示首个失败立即停止。恢复幅度最小性、动态规划与穷举一致性、cost/GAE 手算、未激活统计等由已有专项测试补充：

```bash
python -m unittest examples.test_cascade_s3_recovery examples.test_cascade_p4_context examples.test_cascade_constraint_pipeline examples.test_cascade_training_order -q
```

## 5. 验收范围

本轮先通过 Step 1、Step 2/3，再通过 7 个分支场景。它补充了此前 200 步真实训练回归对罕见分支覆盖不明确的问题。

4 步场景属于分支和接口验收，不用于评价收益、约束长期达标率、收敛性或 P2 临时 0.20 m 参数的工程合理性。P1 硬不可行仅检查显式报错，不伪造后续训练。

后续适合进行新版代码的独立短程训练与完整评估，对比分支频率、原始超限量和 λ 演化；长期正式训练前仍需审定临时参数依据。

本轮补充回归：26 项既有专项测试全部通过，分支脚本语法检查及 git diff --check 通过。
