# AGENTS.md

## 项目说明

本仓库实现的是一个面向五库串联梯级水库联合调度的“双层安全多智能体强化学习”方法。

在修改任何 `cascade_reservoir` 相关代码之前，必须先阅读：

docs/论文与代码统一上下文.md

该文档是本项目论文思路、算法语义、工程约束和代码接口的统一依据。

如果代码实现与该文档存在冲突，应先判断：

1. 是否是代码实现错误；
2. 是否是接口没有同步；
3. 是否是配置或数据缺失；
4. 是否确实需要修改论文方法。

不得为了让代码暂时跑通，而擅自改变论文算法语义。

---

## 固定水力顺序

五座水库顺序固定为：

WDD（乌东德）
  ↓
BHT（白鹤滩）
  ↓
XLD（溪洛渡）
  ↓
XJB（向家坝）
  ↓
THR（三峡）


五座水库对应五个独立 Actor。

不得擅自改为参数共享。

---

## 第一层：当前时刻安全动作层

第一层的固定流程为：

S201 单库局部可行性
    ↓
S202 相邻水库动作兼容性
    ↓
S203 梯级后向可延拓性
    ↓
若联合可行集为空
    ↓
S3 分级最小松弛
    ↓
S4 按水流方向顺序采样


约束优先级固定为：

P1 > P2 > P3 > P4

当正常联合可行集为空时，恢复顺序固定为：


保持 P1 不变
    ↓
最小必要松弛 P4
    ↓
若仍为空且 P4 已达最大松弛
    ↓
最小必要松弛 P3
    ↓
若仍为空且 P3/P4 均已达最大松弛
    ↓
在 P1 安全的完整五库联合动作中
选择 P2 总违反量最小的联合动作


P1 不允许主动松弛。

S201 必须判断当前 `ConstraintContext` 中全部正在生效的 Layer-1 约束。

特别是：

```text
P4 在适用运行阶段（例如 dry_supply）必须参与正常 S201/S202/S203 可行性判断
```

P4 不是“只有空集以后才出现”的约束；正常模式下，只要 P4 当前生效，就应先严格满足。只有完整联合可行集为空时，S3 才允许按照 P4 → P3 → P2 的恢复顺序，在规定范围内对 P4 做最小必要松弛。

S201 本身不负责松弛。约束松弛只能由 S3 控制，并在新的恢复参数下重新进行联合可行性判断。

---

## 当前水力因果关系

当前版本不显式考虑河道传播时滞。

对任一下游水库：


当前总入流 = 直接上游水库当日实际总下泄 +  本库当日区间入流


即：


BHT inflow = WDD Q_exec + BHT interval inflow
XLD inflow = BHT Q_exec + XLD interval inflow
XJB inflow = XLD Q_exec + XJB interval inflow
THR inflow = XJB Q_exec + THR interval inflow


只有 WDD 使用外部总来水。

BHT、XLD、XJB、THR 数据中的来水字段是区间入流，不允许在 DataLoader 中提前累计成总入流。

当前实现必须保持：

Q_exec == Q_target

如果以后引入动作裁剪、投影或二次修正，必须同时修改：


S202
S4 SequentialSampler
CascadeReservoirEnv.step()


不得只修改其中一个模块。

---

## S4 顺序行为采样规则

真实动作采样顺序必须固定为：


WDD
 ↓
BHT
 ↓
XLD
 ↓
XJB
 ↓
THR


禁止先同时计算五个 Actor 的动作，然后再逐库应用 mask。

正确过程必须是：


WDD 采样
    ↓
得到 WDD 当前实际下泄
    ↓
更新 BHT 当前总入流
    ↓
构造 BHT 当前决策观测
    ↓
根据 WDD 已选动作构造 BHT 条件 mask
    ↓
BHT 采样
    ↓
继续向下游传播
```

下游 Actor 只能在直接上游动作确定之后进行真实动作采样。

---

## Observation 和 centralized state

五库基础观测：

每库 17 维

真实行为 Actor observation：

WDD：17 维
BHT：18 维
XLD：18 维
XJB：18 维
THR：18 维


下游 18 维 observation：


17 维 base observation + 1 维当前实际总入流


HARL vector env 接口可以返回统一的：


18 维 placeholder observation × 5


但这些 wrapper observation 不是 Actor 的真实行为观测。

Actor 真正使用的 observation 必须来自：


CascadeReservoirEnv.prepare_step() + SequentialSampler


centralized state 固定为：


5 × 17 = 85 维


即五库 base observation 的拼接。

Reward Critic 和所有 Constraint Critic 必须使用同一个联合动作采样前的 `x_t`。

---

## 行为轨迹 Buffer 规则

对每个 Actor 和时间步 `t`，以下内容必须属于同一次真实行为采样：

decision_observation[t]
action_mask[t]
action[t]
old_log_prob[t]

不得出现 observation/mask 与 action/log_prob 时间索引错位。

未经证明，不得把当前自定义行为轨迹写入逻辑改回普通的通用 `actor_buffer.insert()`。

---

## 首日历史规则

仿真第一天没有可信的前一日实际下泄：

previous_release_m3s = None


因此第一天：

关闭 P3 中的下泄变化子约束

但是：

P3 水位变化约束仍然有效


不得人为构造“前一天的下泄”来方便程序运行。

---

## 第二层：长期约束学习

第二层采用 HAPPO-Lagrangian。

结构为：

```text
1 个 Reward Critic
+
K 个 Constraint Critic
+
K 个拉格朗日乘子 λ
```

每一条实际存在的“水库-约束类型”必须独立维护：

```text
raw_violation
normalization Z_j
tolerance epsilon_tol_j
active_flag chi_j,t
budget d_j
lambda_j
```

长期约束处理：

\[
v_{j,t}
=
\frac{v^{raw}_{j,t}}{Z_j}
\]

\[
\tilde v_{j,t}
=
\max(0,v_{j,t}-\epsilon^{tol}_j)
\]

训练链路必须保留是否实际生效：

\[
c^{train}_{j,t}
=
\chi_{j,t}\tilde v_{j,t}
\]

其中：

```text
chi_j,t = 1
→ 当前时段是第 j 条长期约束的有效样本

chi_j,t = 0
→ 不得把当前时段作为第 j 条长期约束的有效违规样本
```

Reward advantage：

\[
A^R
\]

第 \(j\) 条长期约束 advantage：

\[
A^{C_j}
\]

Actor 更新使用：

\[
A^L = A^R - \sum_j \lambda_j A^{C_j}
\]

一个训练批次内，五个 Actor 更新前先固定：

```text
lambda_snapshot
```

完成本轮 Actor 和 Critic 更新后，再根据第 \(j\) 条约束在当前 batch **实际生效时段**内的平均违反程度更新 \(\lambda_j\)。

如果：

\[
\sum_{t\in\mathcal B}\chi_{j,t}=0
\]

则本 batch：

```text
不统计该约束平均违反程度
lambda_j 保持不变
```

不得在同一个训练批次的五个 Actor 顺序更新中途更新 λ。

不得把“约束未生效”当成“有效的零违规样本”去降低 λ。

---

## 环境动作顺序与 HAPPO 参数更新顺序必须区分

环境真实动作采样顺序固定：

```text
WDD → BHT → XLD → XJB → THR
```

但 HAPPO 参数更新顺序仍然可以是随机顺序：

```python
torch.randperm(num_agents)
```

不得因为水力采样顺序固定，就把 HAPPO 参数更新顺序也强制固定。

同时，HAPPO 不能实现成“随机顺序下五个独立 PPO”。

对轨迹中第 \(i\) 个 Agent，必须基于真实保存的：

```text
decision_observation[t]
action_mask[t]
action[t]
old_log_prob[t]
```

在完全相同的行为条件下计算 new/old policy ratio：

\[
\rho_{i,t}
=
\frac{\pi_{\theta_i}(a_{i,t}\mid o^{dec}_{i,t},M_{i,t})}
{\pi_{\theta_i^{old}}(a_{i,t}\mid o^{dec}_{i,t},M_{i,t})}
\]

若随机更新顺序为：

\[
\sigma=(\sigma_1,\ldots,\sigma_N)
\]

则：

\[
\mu_{0,t}=1
\]

更新第 \(m\) 个 Agent \(\sigma_m\) 时，必须把前面已经完成更新的 Agent 最终 ratio 累乘：

\[
\mu_{m-1,t}
=
\prod_{k=1}^{m-1}\rho_{\sigma_k,t}
\]

并把 \(\mu_{m-1,t}\) 带入当前 Agent 的 HAPPO surrogate。

当前 Agent 更新完成后，要在原轨迹保存的同一 observation / mask / action 条件下重新计算其最终 new/old ratio，再累乘到 \(\mu\) 中供下一个 Agent 使用。

必须保持：

```text
环境行为顺序固定
HAPPO 参数更新顺序随机
已更新 Agent 的 probability ratio 对后续 Agent 更新产生顺序依赖
```

---

## 当前线程限制

当前 `CascadeReservoirRunner` 需要直接访问：

```python
raw_env.prepare_step()
```

因此当前工程实现只支持：

n_rollout_threads = 1
n_eval_rollout_threads = 1
ShareDummyVecEnv


不得直接改为 `ShareSubprocVecEnv`。

如果未来需要多进程，必须先实现 `prepare_step()` 对应的进程通信接口。

---

## 参数和数据规则

不得擅自新增或修改没有依据的物理参数或运行参数，例如：

水位边界
泄量边界
机组能力
长期约束阈值
constraint budget
效率
动作数
P3/P4 参数


查找顺序应为：

1. 项目 YAML
2. 数据文件
3. 已有 Python 配置
4. docs/论文与代码统一上下文.md

如果仍然缺失，必须明确指出缺什么，不能自行猜一个“合理值”。

---

## 模块职责

### `data_loader.py`

只负责：


读取数据
校验数据
提供数据


不得负责：


动作 mask
发电计算
reward
S201/S202/S203
下游总入流累计
策略选择


### `reservoir_physics.py`

负责：

Z-V 转换
水量平衡
下泄拆分
功率
能量
单库 transition
```

### `action_mapping.py`

负责：

离散动作 index
↔
候选总下泄

### `s201_local_feasibility.py`

负责单库局部可行性。

### `s202_compatibility.py`

负责相邻库动作兼容矩阵。

### `s203_extendability.py`

负责梯级后向可延拓性。

### `s3_relaxation.py`

负责：


normal
→ P4 最小松弛
→ P3 最小松弛
→ P2 minimum-violation fallback


### `sequential_sampler.py`

负责：


WDD → BHT → XLD → XJB → THR


的真实行为采样，以及保存真实行为：


decision observation
action mask
action
old log probability
RNN state


### `cascade_reservoir_env.py`

负责：

reset
prepare_step
step
状态转移
reward
constraint cost

### `cascade_reservoir_runner.py`

负责连接：


环境
SequentialSampler
Reward Critic
Constraint Critics
ConstraintBuffer
LagrangianManager
HAPPO


---

## 修改代码时必须保持的关键不变量

### 不变量 1：水力因果


上游动作确定
→ 下游当前总入流确定
→ 下游真实 observation 确定
→ 下游条件 mask 确定
→ 下游 Actor 才能采样


### 不变量 2：安全动作

送入 `env.step()` 的联合动作必须属于当前 `SafetyRecoveryResult` 允许的完整可执行路径。

### 不变量 3：P1 不允许主动松弛

任何恢复机制都不能为了获得可行动作而突破 P1。

### 不变量 4：没有隐藏动作裁剪

当前必须满足：

Q_exec == Q_target


### 不变量 5：同一个 `x_t`

以下必须一致：

```text
prepare_step share_observation
reward critic buffer share_obs[t]
constraint critic buffer share_obs[t]
```

### 不变量 6：行为轨迹同索引

以下必须在同一 `t`：


o_dec[t]
mask[t]
action[t]
old_log_prob[t]

---

### 不变量 7：长期约束生效时段

```text
active_flag[j,t] = 1
→ 才参与第 j 条长期约束的有效统计

整个 batch 无激活样本
→ lambda_j 不更新
```

### 不变量 8：HAPPO 前序 ratio 累乘

```text
随机更新顺序中已更新完成的 Agent
→ 在原行为 observation / mask / action 条件下重新计算最终 new/old ratio
→ 累乘为 mu
→ 用于后续 Agent 的 HAPPO surrogate
```

不得退化为五个彼此独立的 PPO 更新。

---

## 修改纪律

每次 Codex 任务都必须：

1. 先阅读 `AGENTS.md`；
2. 再阅读 `docs/论文与代码统一上下文.md`；
3. 再读取本次任务相关真实代码；
4. 不根据文件名猜接口；
5. 一次只解决一个明确问题；
6. 做最小必要修改；
7. 不改变论文算法语义来绕过报错；
8. 修改后运行语法检查和相关测试；
9. 最后报告修改文件、原因、测试结果和剩余问题。

---

## 禁止用以下方式“修复”程序

```text
mask 为空 → 直接改成全 1
P1 不可行 → 自动扩大 P1
shape 不匹配 → 随便 pad/truncate
下游入流不好算 → 使用昨天的值
buffer 错位 → 忽略 available_actions
constraint critic 报错 → 暂时删除 constraint critic
长期约束未生效 → 当成有效零违规样本更新 lambda
HAPPO 顺序更新麻烦 → 改成五个独立 PPO
缺少参数 → 自己编一个默认值
```

这些行为虽然可能让代码暂时运行，但会改变论文方法。

---

## 集成测试顺序

必须按以下顺序逐层测试：

```text
Step 1
reset
→ prepare_step

Step 2
prepare_step
→ collect
→ Reward Critic
→ Constraint Critics
→ SequentialSampler

Step 3
collect
→ env.step

Step 4
env.step
→ insert

Step 5
rollout
→ compute

Step 6
compute
→ train
→ 检查 constraint active_flag / effective violation
→ 检查 HAPPO preceding-agent ratio accumulation
→ 检查无激活 constraint 的 lambda 保持不变

Step 7
完整链路
reset
→ prepare_step
→ collect
→ env.step
→ insert
→ compute
→ train
```

不要跳过失败步骤继续向后。

---

## Codex 每次任务的推荐提示

```text
请先阅读：

1. AGENTS.md
2. docs/论文与代码统一上下文.md

然后只处理下面这个任务，不要扩大修改范围。

【任务】
<明确描述当前错误或待实现接口>

【相关文件】
- <文件 1>
- <文件 2>

【必须保持】
- 五库顺序 WDD → BHT → XLD → XJB → THR
- P1 不允许主动松弛
- 下游总入流 = 当日直接上游实际下泄 + 本库区间入流
- 当前 Q_exec == Q_target
- 下游 Actor 必须等直接上游动作确定后再构造 observation/mask 并采样
- 五个 Actor 独立
- 不得创造没有依据的新参数

【完成标准】
1. 先检查现有接口，不要猜。
2. 做最小必要修改。
3. 不改变论文算法。
4. 运行相关测试。
5. 最后说明修改了什么、测试结果、还有什么问题。
```
