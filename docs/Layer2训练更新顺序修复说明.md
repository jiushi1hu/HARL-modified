# Layer 2 训练更新顺序修复说明

依据：《论文与代码统一上下文_严谨修订版.md》第 14、16、17、19、21 节。

## 1. 本轮问题

原 `CascadeReservoirRunner.train()` 在算好优势后，先更新五个 Actor，再更新 Reward/Constraint Critics，最后更新 λ。这与修订版要求的 Critics 在先、Actors 在后不一致。

另一个与顺序更新直接相关的问题：累计前序 Agent 概率比时，旧实现用更新前重新评估的概率作分母。现在统一使用行为 Buffer 中保存的 `action_log_probs`，与 HAPPO 单个 Actor 更新时的分母一致，不依赖“重新评估一定等于原行为概率”的假设。

## 2. 修改后的执行步骤

```text
compute 已得到 rollout 对应的 returns / value targets
  ↓
使用 rollout 的旧 value_preds、更新前的 ValueNorm 状态计算 A_R、A_C
  ↓
固定 lambda_snapshot
  ↓
计算并保留 A_L = A_R − Σ lambda_snapshot[j] × A_C[j]
  ↓
更新 Reward Critic
  ↓
逐一更新全部 Constraint Critics
  ↓
随机生成五个 Actor 的参数更新顺序
  ↓
mu = 1
对每个 Actor：
    将当前 mu 写入该 Actor Buffer 的 factor
    用本轮固定 A_L 的副本更新 Actor
    用原轨迹 observation / mask / action 重算更新后的 log_prob
    rho_final = exp(new_log_prob − 保存的行为 old_log_prob)
    mu = mu × rho_final
  ↓
按约束实际生效样本统计更新 λ
```

关键点：

- Critic 训练可能更新自己的 ValueNorm，故 A_L 必须在任何 Critic 更新之前算好。不能把优势计算代码一起移到 Critic 更新之后。
- 每个 Actor 接收 A_L 的独立副本，防止内部标准化等操作影响后续 Actor。
- `lambda_snapshot` 在本轮始终固定；所有 Actor 完成后才调用乘子更新。
- 整批无激活样本时，对应 λ 不更新。
- HAPPO 的 `factor` 是已完成更新 Actor 的最终 ratio 累乘，不是五个独立 PPO。
- 环境行为采样仍按 WDD→BHT→XLD→XJB→THR，训练参数更新顺序仍随机。
- 日志的 Actor 结果仍按水库 ID 存放，避免随机更新顺序打乱水库名称与指标的对应。

## 3. 修改文件

| 文件 | 作用 |
|---|---|
| `harl/runners/cascade_reservoir_runner.py` | Critics 前移到 A_L 固定之后；累计 ratio 使用已保存的行为概率 |
| `examples/test_cascade_training_order.py` | 更新顺序、优势固定、非单位 ratio、未激活 λ 及指标身份测试 |
| `examples/test_cascade_training_integration.py` | 真实网络逐个 Critic/Actor/λ 更新顺序和累计 ratio 回归 |

没有修改物理参数、Layer 1 安全规则、动作数、预算、学习率或 AGENTS.md。

## 4. 验收方法

专项测试使用人工构造的训练数组，不改生产参数：

1. 记录事件，要求 Reward Critic → Constraint Critics → 指定随机顺序的五个 Actor → λ。
2. 模拟 Critic 训练改变归一化状态与数据，检查 Actor 仍得到原 A_L。
3. 模拟某 Actor 修改收到的优势数组，检查后续 Actor 不受污染。
4. 使用不同 Actor、不同时间步的非单位 ratio，逐项检查下一 Actor 收到的 mu。
5. 给行为 log probability 设置非零值，确保分母来自 Buffer。
6. 分别验证有效约束更新 λ、整批未激活时 λ 不变。

真实网络回归按顺序执行：

```bash
conda activate harl
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONDONTWRITEBYTECODE=1
python -m unittest examples.test_cascade_training_order -v
python -m examples.test_cascade_reservoir
python -m examples.test_cascade_reservoir_step3
python -m examples.test_cascade_training_integration
```

200 步集成测试除原有轨迹索引、Critic 状态同步、compute/train 和检查点恢复检查外，还验证：

- 真实 Reward Critic 和每个 Constraint Critic 均先于 Actor 更新。
- 所有 Actor 使用更新前固定的优势和 λ。
- 至少存在非单位的最终概率比，且其累乘传给后续 Actor。
- λ 更新事件最后发生。

## 5. 范围与后续工作

本轮只修复训练更新顺序及相关概率比契约，不表示整个 Layer 2 已完全符合修订版。
后续已补齐原始违反量 xi、active_flag 的完整保存与 cost 重建，规范 beta 兼容和折扣统计分子；详见《Layer2原始违反量与训练cost链路修复说明.md》。

已运行的训练进程不会自动加载本轮修改。新启动进程才使用新代码；本轮测试不代表长期训练已收敛。

## 6. 本轮结果

4 项训练专项测试通过；Step 1、Step 2/3、200 步真实网络集成回归按顺序通过。集成测试确认 Critics → Actors → λ、固定 A_L、非单位概率比累乘，以及五个 Actor、六个 Critic 和 λ 的检查点保存恢复。语法检查和 git diff --check 通过。
