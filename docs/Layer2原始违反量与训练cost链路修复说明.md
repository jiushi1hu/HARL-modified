# Layer 2 原始违反量与训练 cost 链路修复说明

依据：严谨修订版第 11、12、17、19、21 节。本轮修复原始违反量从环境到训练的保存与转换链路。

## 1. 原先的问题

- 环境输出 ξ，但 Runner 提取反馈时丢弃它，ConstraintBuffer 只有 cost 和 active_flag。
- compute 直接使用已经写入的 cost，无法从轨迹中重新核对原始违反量的转换。
- 环境对未生效约束直接跳过计算，原始 ξ 被记成零，难以与有效零违反区分。
- 原始违反公式允许额外 beta 倍率；当前配置虽均为 1，接口仍允许偏离修订版公式。
- 折扣统计分子又乘一次 active_flag。二值标记下结果相同，但表达重复，不符合文档定义。

## 2. 当前执行流程

```text
实际执行下泄/出力
→ ξ = sqrt(max(0, (requirement − actual) / requirement))
→ c = ξ / z
→ χ = 当前运行阶段是否生效
→ v = χ × max(0, c − tolerance)
→ env.step infos 输出 ξ、χ、v 和约束 ID
→ Runner 校验各库反馈一致、约束 ID 顺序、shape、数值、二值 χ 和转换公式
→ insert 在同一个 t 保存 ξ、χ、v
→ compute_returns 先从保存的 ξ、χ 和固定 z/tolerance 重建 v
→ 每个 Constraint Critic 使用 v 计算 GAE/returns
→ Critics、Actors 更新结束后按有效样本统计更新 λ
```

原始 ξ、归一化 c、训练 v 都是无量纲量。生态/航运使用实际总下泄，保证出力使用实际出力。

未生效时仍计算和保存原始 ξ、c，作为诊断；v 必须为零，χ 必须为零。原始 ξ 非零不代表该时段是有效违规样本。统计论文中的违规率时也必须结合 χ。

`beta` / `scale_coefficient` 仅兼容旧配置的 1.0，非 1 值明确报错，原始公式不再乘它。没有改变现有 requirement、z、tolerance、budget、学习率或物理参数。

## 3. Buffer 数据契约

`raw_violations[t, thread, j]`、`active_flags[t, thread, j]`、`costs[t, thread, j]` 属于同一次环境动作执行结果。

- z、tolerance 按 constraint_id 顺序从已校验的环境 ConstraintSpec 传入，不猜默认值。
- 写入时检查原始量非负有限、生效标记为 0/1；检查环境 v 与公式一致。
- 环境和 Buffer 浮点计算的核对容差为 rtol=1e-6、atol=1e-7；这仅用于数值一致性检查，不是新增运行容忍阈值。未生效项的环境 cost 必须严格为零。
- 原始数组复制进 Buffer，环境后续修改反馈数组不能覆盖已存轨迹。
- Buffer 完成一批、step 回到零时，仍保留整批 ξ/χ，直到 compute/train 完成。
- `after_update()` 清空 ξ、χ、v，防止上一批约束数据残留；共享状态/RNN 的批次衔接沿用原 Critic Buffer 机制。
- compute 重建 v 时不修改 ξ、χ，也不重建行为 observation、mask、action 或 log_prob。

折扣有效均值严格为：

```text
分子 = Σ gamma^t × v[t,j]
分母 = Σ gamma^t × χ[t,j]
```

分母为零时统计标记无效，均值保留 NaN；LagrangianManager 不更新对应 λ，不把它当作零均值降低 λ。

## 4. 修改文件

| 文件 | 作用 |
|---|---|
| `harl/envs/cascade_reservoir/constraint_costs.py` | 原始公式、未激活原始量保留、v 生效处理、旧 beta 校验 |
| `harl/runners/cascade_reservoir_runner.py` | 传入 z/tolerance、提取与校验 ξ/χ/v、同索引 insert |
| `harl/common/buffers/constraint_buffer.py` | 保存 ξ、重建训练 cost、折扣统计、批次清理 |
| `harl/configs/envs_cfgs/cascade_reservoir.yaml` | 只修正 beta 的说明注释，本轮未改参数值 |
| `examples/test_cascade_constraint_pipeline.py` | 原始公式、未激活行为、索引、cost、手算 GAE、有效统计、输入错误测试 |
| `examples/test_cascade_training_integration.py` | 实际轨迹逐步核对 ξ/χ/v；compute 后核对原始量不变及 cost 重建 |

## 5. 验证方式

```bash
conda activate harl
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONDONTWRITEBYTECODE=1
python -m unittest examples.test_cascade_constraint_pipeline examples.test_cascade_training_order -v
python -m examples.test_cascade_reservoir
python -m examples.test_cascade_reservoir_step3
python -m examples.test_cascade_training_integration
```

专项测试使用非默认 z/tolerance、混合激活时段、整批未激活约束，避免仅在 z=1、tolerance=0、全部激活的生产配置下验证。
手算样例核对 GAE/returns、折扣分母、生效计数，确认未激活时段不会稀释约束平均值。

本轮没有修改 AGENTS.md，也没有操作已有训练进程。当前结果属于代码与接口验证，不能据此声称长期训练收敛或工程参数已有论证。

## 6. 本轮验收结果

8 项相关专项测试通过；Step 1、Step 2/3、200 步真实网络集成回归按顺序通过。验证了 ξ/χ 同时间步存储、训练 cost 重建、GAE/returns、Critics→Actors→λ 更新顺序、固定优势、非单位 ratio 累乘，以及模型和 λ 保存恢复。语法检查、git diff --check 通过。尚未执行新的长期训练及全部恢复分支的逐分支完整训练验收。
