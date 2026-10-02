五库串联梯级水库双层安全多智能体强化学习方法

本文档用于统一乌东德、白鹤滩、溪洛渡、向家坝和三峡五库串联系统的研究方法、实例化设置、数据要求、参数配置、程序职责以及训练验证要求。

本文档分为五部分。第一部分给出核心方法；第二部分给出五库实例的数据、CSV
字段和参数设置；第三部分给出代码实现、接口与训练验证要求；第四部分列出尚未确定、必须通过真实数据或配置补充的项目，以及修改程序时必须遵守的一致性边界；第五部分给出一些具体的参数的取值。

一、研究对象、基本假设与总体框架

研究对象为金沙江---长江干流上的五座串联梯级水库，按水流方向依次为：

乌东德 WDD → 白鹤滩 BHT → 溪洛渡 XLD → 向家坝 XJB → 三峡 THR

五座水库分别对应五个协作智能体。五个 Actor
独立设置，不共享策略参数；系统共享梯级发电奖励，但不同水库可以承担不同的生态下泄、保证出力和航运等长期运行约束。

当前研究采用日尺度调度，时间步长为

\[ `\Delta`{=tex}t=86400 {`\rm s`{=tex}}. \]

时间区间设为 2023-01-01 至
2025-12-31。不单独考虑相邻水库之间的河道传播时滞，因此在同一决策日内，下游水库的实际入流由直接上游水库当日实际总下泄量与两库之间的区间入流共同构成。

本研究采用"双层安全"结构。第一层负责当前决策步的联合可行性，核心是 S1 至
S4：建立模型、计算单库局部可行性、建立相邻候选动作兼容关系、从下游向上游递推完整路径可延拓性、在联合可行集为空时进行分级最小松弛，并最终按水流方向进行条件动作采样。第二层负责长期策略训练，即
S5：利用集中式 Reward Critic、各长期约束独立的 Constraint Critic、HAPPO
策略更新与拉格朗日乘子实现发电收益和长期运行约束之间的协调优化，称为HAPPO-Lagrangian方法。

第一层与第二层不能相互替代。

二、核心方法

2.1 S1：约束合作型马尔可夫博弈

设水库总数为 (N=5)，水库及智能体索引集合为

\[ `\mathcal`{=tex}R={1,2,`\ldots`{=tex},N}. \]

系统状态由每座水库的动态运行信息与固定物理参数共同构成：

\[
s_t={x\^{dyn}\_{i,t},`\kappa`{=tex}*i}*{i`\in`{=tex}`\mathcal`{=tex}R}.
\]

动态信息至少应包括当前水位、库容、当前可获得的外部或区间来水信息、上一时刻实际下泄量、上一时刻出力以及运行阶段信息。固定参数包括库容---水位关系、装机容量、总泄流能力以及调度计算需要的其他物理参数。

联合动作开始采样之前，第 (i) 个智能体可获得基础观测
(o\^{base}\_{i,t})。基础观测由本库局部信息、直接上游信息、直接下游信息和梯级系统信息构成。最上游乌东德在动作采样前已经知道当前外部入流，因此其决策观测直接取基础观测：

\[ o\^{dec}*{1,t}=o\^{base}*{1,t}. \]

对于其余下游水库
(i=2,`\ldots`{=tex},N)，只有在直接上游水库动作确定后，才能得到当前真实入流：

\[ I\_{i,t}=Q\^{exec}*{i-1,t}+I\^{int}*{i,t}, \]

随后构造实际用于 Actor 采样的决策观测

\[ o\^{dec}*{i,t}=(o\^{base}*{i,t},I\_{i,t}). \]

因此，下游基础观测和决策观测不是同一个意思。训练阶段重新计算策略概率时，必须使用行为采样时真实保存的
(o\^{dec}\_{i,t})，不得重新构造另一套观测替代原行为条件。

各水库采用离散总下泄动作。若候选动作索引为

\[ k`\in`{=tex}{0,1,`\ldots`{=tex},N_a-1}, \]

则归一化索引为

\[ u(k)=`\frac{k}{N_a-1}`{=tex}. \]

第 (i) 座水库的候选动作映射为目标总下泄量

\[ Q\^{tar}\_{i,t}(a_i) = Q\^{map,min}\_i+
`\left`{=tex}(Q\^{map,max}\_i-Q\^{map,min}\_i`\right`{=tex})`\psi`{=tex}\_i(u(k)),
\]

其中 (`\psi`{=tex}\_i) 必须严格单调递增，并满足
(`\psi`{=tex}\_i(0)=0)、(`\psi`{=tex}\_i(1)=1)。动作编码范围
(Q\^{map,min}\_i,Q\^{map,max}\_i)
只定义离散动作映射范围，不等同于当前时刻由水位、库容、泄流能力和运行约束决定的动态可行范围。

水量平衡为

\[ V\_{i,t+1} = V\_{i,t} +
`\left`{=tex}(I\_{i,t}-Q\^{exec}\_{i,t}`\right`{=tex})`\Delta`{=tex}t.
\]

若库容---水位关系记为 (V_i=f_i(h))，则下一时刻水位为

\[ h\_{i,t+1}=f_i\^{-1}(V\_{i,t+1}). \]

实际总下泄由过机流量与非发电下泄构成：

\[ Q^{exec}*{i,t}=Q\^{gen}*{i,t}+Q^{non}\_{i,t}. \]

在不存在强制非发电泄放、机组限制或其他特殊调度要求时，总下泄优先分配为过机流量，超过当前有效过机能力的部分作为非发电下泄。当前出力可按

\[ P\_{i,t} = `\min`{=tex} `\left`{=tex}( P_i\^{inst},
`\eta`{=tex}*i`\rho`{=tex}*w gH*{i,t}Q\^{gen}*{i,t} `\right`{=tex}) \]

计算，当前时段发电量为

\[ E\_{i,t}=P\_{i,t}`\Delta`{=tex}t. \]

系统共享发电奖励为

\[ r_t= `\frac{\sum_{i\in\mathcal R}E_{i,t}}{E^{ref}}`{=tex}, \]

其中 (E\^{ref}) 是固定归一化尺度。

2.2 S2：梯级联合可行性动作构造

S2 由 S201、S202 和 S203
三个连续步骤构成。其目标不是分别判断五座水库是否各自存在可行动作，而是在
Actor 采样前证明至少存在一条从乌东德一直延拓到三峡的完整候选动作路径。

S201 在给定入流条件下判断单库局部可行性。设下一时刻允许水位范围为

\[ h\^{min}*{i,t+1} `\le`{=tex} h*{i,t+1} `\le`{=tex} h\^{max}\_{i,t+1},
\]

根据库容---水位关系得到对应库容边界 (V\^{low}*{i,t+1}) 与
(V\^{high}*{i,t+1})。若当前物理总下泄能力范围为
(\[Q\^{min}*{i,t},Q\^{max}*{i,t}\])，则给定入流 (I\_{i,t})
下的动态安全下泄范围为

\[ Q\^{safe,min}*{i,t} = `\max`{=tex} `\left`{=tex}( Q\^{min}*{i,t},
I\_{i,t}+ `\frac{V_{i,t}-V^{high}_{i,t+1}}{\Delta t}`{=tex}
`\right`{=tex}), \]

\[ Q\^{safe,max}*{i,t} = `\min`{=tex} `\left`{=tex}( Q\^{max}*{i,t},
I\_{i,t}+ `\frac{V_{i,t}-V^{low}_{i,t+1}}{\Delta t}`{=tex}
`\right`{=tex}). \]

候选动作的局部可行性定义为

\[ M\^{local}\_{i,t}(a_i) = `\mathbf 1`{=tex}
`\left[ Q^{safe,min}_{i,t} \le Q^{tar}_{i,t}(a_i) \le Q^{safe,max}_{i,t} \right]`{=tex}.
\]

除上述水位和泄流能力外，S201 还必须检查当前 ConstraintContext
中实际生效的 P1、P2、P3、P4 第一层约束。S201
只负责判断，不负责松弛；只有 S3
可以调整允许松弛的运行约束，然后重新执行完整 S201→S202→S203。

S202 建立相邻水库候选动作兼容关系。对于下游水库
(i=2,`\ldots`{=tex},N)，上游候选动作尚未真正执行，因此使用目标总下泄构造候选入流：

\[ I\^{cand}*{i,t}(a*{i-1}) = Q\^{tar}*{i-1,t}(a*{i-1}) +
I\^{int}\_{i,t}. \]

该候选入流只用于当前决策步的动作兼容性判断，不能写回环境物理状态，也不能当作真实入流。将该候选入流交给
S201，若下游候选动作 (a_i) 在此条件下局部可行，则

\[ C\_{i,t}(a\_{i-1},a_i)=1, \]

否则取 0。五库系统依次构造 WDD--BHT、BHT--XLD、XLD--XJB、XJB--THR
四个相邻候选动作兼容矩阵。

S203 从最下游向最上游递推下游可延拓性。定义 (b\_{i,t}(a_i)) 表示候选动作
(a_i) 是否还能继续沿兼容动作路径延拓至最下游。对三峡定义

\[ b\_{N,t}(a_N)=1. \]

然后对 (i=N-1,`\ldots`{=tex},1) 逆向递推：

\[ b\_{i,t}(a_i) = `\mathbf 1`{=tex}
`\left[ \exists a_{i+1}: C_{i+1,t}(a_i,a_{i+1})=1 \land b_{i+1,t}(a_{i+1})=1 \right]`{=tex}.
\]

最上游乌东德的根安全集合还必须同时满足本库局部可行：

\[ A\^{safe}*{1,t} = { a_1`\in`{=tex}A_1: M\^{local}*{1,t}(a_1)=1
`\land`{=tex} b\_{1,t}(a_1)=1 }. \]

对中间水库，在给定已经选择的直接上游动作 (a\_{i-1})
后，只保留"与该上游动作兼容且仍可延拓到三峡"的动作。三峡只需满足与向家坝已选动作兼容。Actor
实际使用的是这一梯级联合可行性 mask，而不是单库 local mask。

2.3 S3：联合可行集为空时的分级最小松弛

参与当前步联合可行性判断的约束按优先级分为

\[ P1\>P2\>P3\>P4. \]

P1
为不可主动放宽的绝对物理安全约束，包括水量平衡、物理泄流能力、库容或水位绝对物理安全边界、设备不可突破能力以及触发紧急安全条件后必须执行的强制泄洪要求。P2
为防洪期、蓄水期和消落期等不同运行阶段的时变水位控制要求。P3
为相邻时段水位变化、下泄变化等平滑性要求。P4
为枯水期泄流与入流关系要求。

若 (A\^{safe}\_{1,t}) 非空，则直接进入 S4；若为空，则保持 P1
不变，并严格按照

\[ P4`\rightarrow`{=tex}P3`\rightarrow`{=tex}P2 \]

的顺序进行恢复。每一级只能在预设最大允许松弛范围内逐步放宽，并且每一个候选松弛幅度都必须重新运行完整的
S201→S202→S203。只要恢复至少一条完整梯级联合可行路径，立即停止继续松弛当前等级，也不进入更高优先级等级。若
P4 达到最大允许松弛仍无法恢复，则处理 P3；若 P3
也达到最大仍无法恢复，再处理 P2。

只有当 P2、P3、P4
均达到各自允许的最大松弛范围而梯级联合可行集仍为空时，才进入最终
fallback：在满足 P1
的候选联合动作中选择运行约束违反程度最小的联合动作，并记录对应违反量。专利没有规定不同运行约束违反量的具体归一化和聚合公式，因此该标量化方式必须由项目配置显式给出，不能自行假设。

若连满足 P1
的候选联合动作都不存在，应视为硬不可行或紧急异常状态，程序必须显式报告，不能把动作
mask 修改为全 1，也不能主动扩大 P1。

2.4 S4：沿水流方向进行真实条件动作采样

真实行为采样顺序固定为

WDD → BHT → XLD → XJB → THR。

乌东德先依据 (o\^{dec}*{WDD,t}) 和根安全 mask
采样动作，得到其当前实际总下泄 (Q\^{exec}*{WDD,t})。白鹤滩随后计算

\[ I\_{BHT,t} = Q\^{exec}*{WDD,t} + I\^{int}*{BHT,t}, \]

形成新的决策观测，并使用与乌东德已选动作相对应的条件安全 mask
采样。溪洛渡、向家坝和三峡依次重复这一过程。

若第 (i) 个 Actor 对候选动作 (a_i) 输出偏好 (z\_{i,t}(a_i))，当前实际
mask 为 (M\_{i,t}(a_i))，则只在 (M\_{i,t}=1) 的动作集合上归一化。mask 为
0 的动作概率必须为 0。程序实现可以使用稳定的 masked-logit
方式，但最终概率语义必须与上述受限 softmax 一致。

全部动作确定后形成梯级联合动作

\[ a_t=(a\_{1,t},a\_{2,t},`\ldots`{=tex},a\_{N,t}). \]

环境执行联合动作，更新库容、水位、过机流量、非发电下泄、出力、发电量和下一时刻状态，并计算系统发电奖励及各长期运行约束原始违反量。

每个智能体必须保存真实采样时的四个核心行为量：

\[ o\^{dec}*{i,t},`\qquad`{=tex} M*{i,t},`\qquad`{=tex}
a\_{i,t},`\qquad`{=tex} `\log`{=tex}p\^{old}\_{i,t}. \]

它们必须属于同一个时间索引和同一次行为采样。训练时不得重新生成下游
observation 或 mask 来替换这些真实行为条件。

当前五库代码约定在动作确定后不再额外进行动作投影、二次裁剪或执行层修正，因此

\[ Q\^{exec}*{i,t}=Q\^{tar}*{i,t}. \]

虽然当前数值相等，符号仍保留区分，因为二者分别表示"候选/目标下泄"和"动作确定后的真实执行下泄"。若后续增加强制泄洪、设备故障修正或其他执行层处理，使二者不再相等，必须整体重新检查
S202 候选兼容性和 S4 真实执行之间的一致性。

2.5 S5：HAPPO-Lagrangian 长期策略优化

每个水库设置独立 Actor；集中式 Critic
使用联合动作采样前已经获得的全局信息 (x_t)
作为输入。当前实例将五座水库的 base observation 拼接为全局信息。集中式
Critic 输出一个系统发电 Reward
Value，以及每一项实际存在的"水库---约束类型"对应的独立 Constraint
Value。

长期约束按具体"水库---约束类型"逐项建立。例如同一水库同时承担生态和保证出力任务时，应建立两个独立约束项；未承担某项任务的水库不建立该约束。

生态下泄原始违反量定义为

\[ `\xi`{=tex}\^{eco}\_{i,t} =
`\sqrt{ \max \left( 0, \frac{Q^{eco}_i-Q^{exec}_{i,t}}{Q^{eco}_i} \right) }`{=tex}.
\]

保证出力原始违反量为

\[ `\xi`{=tex}\^{gua}\_{i,t} =
`\sqrt{ \max \left( 0, \frac{P^{gua}_i-P_{i,t}}{P^{gua}_i} \right) }`{=tex}.
\]

航运下泄原始违反量为

\[ `\xi`{=tex}\^{nav}\_{i,t} =
`\sqrt{ \max \left( 0, \frac{Q^{nav}_i-Q^{exec}_{i,t}}{Q^{nav}_i} \right) }`{=tex}.
\]

平方根用于增强约束边界附近小幅违反之间的区分度，同时压缩严重违反样本之间的数值差异。

对第 (j) 项长期运行约束，定义生效指标

\[ `\chi`{=tex}\_{j,t}`\in`{=tex}{0,1}. \]

原始违反量 (`\xi`{=tex}\_{j,t}) 先按固定尺度 (z_j) 归一化：

\[ c\_{j,t}=`\frac{\xi_{j,t}}{z_j}`{=tex}, \]

再结合容忍阈值 (`\epsilon`{=tex}\^{tol}\_j) 和生效指标得到真正用于
Constraint Critic 和策略优化的约束代价：

\[ v\_{j,t} = `\chi`{=tex}*{j,t} `\max`{=tex} `\left`{=tex}( 0,
c*{j,t}-`\epsilon`{=tex}\^{tol}\_j `\right`{=tex}). \]

当 (`\chi`{=tex}\_{j,t}=0) 时，该时刻对应约束不产生训练压力。

Reward Critic 和各 Constraint Critic 分别计算 TD 误差与
GAE，得到发电优势 (A_t\^R) 和各约束优势 (A_t\^{C_j})。在一次 Actor
更新轮内固定当前拉格朗日乘子，构造

\[ A_t\^L = A_t\^R -
`\sum`{=tex}\_{j=1}\^{K}`\lambda`{=tex}\_jA_t\^{C_j}. \]

真实动作采样顺序固定为水流方向，但 HAPPO 的 Actor
参数更新顺序必须在已采集轨迹上随机生成，二者互不等同。对每个 Actor
计算新旧策略概率比时，必须使用 S4 中已经保存的同一个 (o\^{dec})、mask 和
action。已更新 Actor 的最终 new/old ratio 需要累乘到后续 Actor 的 HAPPO
surrogate 中，不能退化为随机顺序的五次独立 PPO。

每项长期约束拥有独立预算 (d_j)、拉格朗日乘子 (`\lambda`{=tex}*j)
以及乘子步长
(`\eta`{=tex}*{`\lambda`{=tex}\_j})。若当前训练批次内该约束存在有效生效样本，则按实际生效时段统计其折扣加权平均约束代价
(`\hat`{=tex}J_j)，并更新

\[ `\lambda`{=tex}\_j `\leftarrow`{=tex} `\max`{=tex} `\left`{=tex}( 0,
`\lambda`{=tex}*j+ `\eta`{=tex}*{`\lambda`{=tex}\_j}
(`\hat`{=tex}J_j-d_j) `\right`{=tex}). \]

若整个 batch 内该约束完全未生效，则不计算
(`\hat`{=tex}J_j)，也不更新对应
(`\lambda`{=tex}\_j)。未生效不能当成有效的零违规样本。

2.6 训练目标、优势函数与参数更新定义

为保证 HAPPO-Lagrangian 方法能够被完整复现，需要进一步明确 Reward
Critic、Constraint
Critic、GAE、策略更新以及拉格朗日乘子更新中的数学定义。

2.6.1 Reward Value 与 Reward GAE

系统共享发电奖励定义为：

\[ r_t=`\frac{\sum_{i\in\mathcal R}E_{i,t}}{E^{ref}}`{=tex} \]

其中：

  符号                   定义
  ---------------------- ---------------------------------
  (r_t)                  第 (t) 个调度时段的系统发电奖励
  (E\_{i,t})             第 (i) 座水库当前时段发电量
  (E\^{ref})             奖励归一化参考尺度
  (`\mathcal `{=tex}R)   水库智能体集合

发电功率定义为：

\[ P\_{i,t} = `\min`{=tex} `\left`{=tex}( P_i\^{inst},
`\eta`{=tex}*i`\rho`{=tex}*w g H*{i,t}Q\^{gen}*{i,t} `\right`{=tex}) \]

其中：

  符号                定义
  ------------------- ---------------------------
  (P\_{i,t})          第 (i) 座水库当前发电功率
  (P_i\^{inst})       装机容量
  (`\eta`{=tex}\_i)   综合发电效率
  (`\rho`{=tex}\_w)   水密度
  \(g\)               重力加速度
  (H\_{i,t})          有效水头
  (Q\^{gen}\_{i,t})   发电流量

当前时段发电量：

\[ E\_{i,t}=P\_{i,t}`\Delta `{=tex}t \]

Reward Critic 用于估计系统长期累计收益：

\[ V\^R(x_t) = `\mathbb `{=tex}E `\left[
\sum_{l=0}^{\infty}
\gamma^l r_{t+l}
\right]`{=tex}\]

其中：

  `\符号`{=tex}      定义
  ------------------ ------------------------------
  (V\^R(x_t))        Reward Critic 输出的状态价值
  (`\gamma`{=tex})   折扣因子
  (x_t)              集中式 Critic 输入状态

Reward TD 误差定义为：

\[ `\delta`{=tex}*t\^R = r_t+ `\gamma `{=tex}V\^R(x*{t+1}) - V\^R(x_t)
\]

采用 GAE 计算奖励优势：

\[ A_t\^R = `\sum`{=tex}*{l=0}\^{T-t-1}
(`\gamma`{=tex}`\lambda`{=tex}*{GAE})\^l `\delta`{=tex}\_{t+l}\^R \]

其中：

  `\符号`{=tex}              定义
  -------------------------- ----------------------
  (`\lambda`{=tex}\_{GAE})   GAE 平滑参数
  \(T\)                      当前 trajectory 长度

------------------------------------------------------------------------

2.6.2 Constraint Cost 与 Constraint GAE

第 (j) 项长期约束代价：

\[ v\_{j,t} = `\chi`{=tex}*{j,t} `\max`{=tex}
(0,c*{j,t}-`\epsilon`{=tex}\_j\^{tol}) \]

其中：

\[ c\_{j,t}=`\frac{\xi_{j,t}}{z_j}`{=tex} \]

符号定义：

  符号                           定义
  ------------------------------ ---------------------
  \(j\)                          长期约束编号
  (`\xi`{=tex}\_{j,t})           第 (j) 项原始违反量
  (z_j)                          归一化尺度
  (`\epsilon`{=tex}\_j\^{tol})   约束容忍阈值
  (`\chi`{=tex}\_{j,t})          约束生效标志
  (v\_{j,t})                     用于训练的约束代价

Constraint Critic 估计未来累计约束违反：

\[ V\^{C_j}(x_t) = `\mathbb `{=tex}E `\left[
\sum_{l=0}^{\infty}
\gamma^l v_{j,t+l}
\right]`{=tex}\]

对应 TD 误差：

\[ `\delta`{=tex}*t\^{C_j} = v*{j,t} +
`\gamma `{=tex}V\^{C_j}(x\_{t+1}) - V\^{C_j}(x_t) \]

约束优势：

\[ A_t\^{C_j} = `\sum`{=tex}*{l=0}\^{T-t-1}
(`\gamma`{=tex}`\lambda`{=tex}*{GAE})\^l `\delta`{=tex}\_{t+l}\^{C_j} \]

每一个实际存在的"水库---约束类型"均独立计算对应的 Constraint Value 和
Advantage。

------------------------------------------------------------------------

2.6.3 Lagrangian Advantage

在一次 Actor 更新过程中，固定当前拉格朗日乘子：

\[ A_t\^L = A_t\^R - `\sum`{=tex}\_{j=1}\^{K} `\lambda`{=tex}\_j
A_t\^{C_j} \]

其中：

  符号                   定义
  ---------------------- ---------------------------
  (A_t\^L)               拉格朗日修正后的策略优势
  \(K\)                  长期约束总数量
  (`\lambda`{=tex}\_j)   第 (j) 项约束拉格朗日乘子

该形式表示：

-   Reward Advantage 提升发电收益；
-   Constraint Advantage 抑制长期约束违反。

------------------------------------------------------------------------

2.6.4 HAPPO Actor 更新

第 (i) 个智能体策略概率比：

\[ r_i(`\theta`{=tex}\_i) = `\frac{
\pi_{\theta_i}(a_i|o_i^{dec})
}{
\pi_{\theta_i^{old}}(a_i|o_i^{dec})
}`{=tex} \]

其中：

  符号                                        定义
  ------------------------------------------- --------------------------
  (`\pi`{=tex}\_{`\theta`{=tex}\_i})          当前 Actor 策略
  (`\pi`{=tex}\_{`\theta`{=tex}\_i\^{old}})   旧策略
  (o_i\^{dec})                                真实采样时保存的决策观测

HAPPO/PPO 截断目标：

\[ L_i(`\theta`{=tex}\_i) = `\mathbb `{=tex}E `\left[
\min
\left(
r_i(\theta_i)A_t^L,
clip(r_i(\theta_i),1-\epsilon,1+\epsilon)A_t^L
\right)
\right]`{=tex}\]

其中：

  `\符号`{=tex}        定义
  -------------------- ---------------
  (`\epsilon`{=tex})   PPO clip 范围

训练过程中：

-   真实动作采样顺序固定为 WDD→THR；
-   Actor 参数更新顺序随机生成；
-   后续 Actor 更新必须累计前序已更新 Actor 的 ratio。

------------------------------------------------------------------------

2.6.5 Critic Loss

Reward Critic 使用：

\[ L\^R = `\frac12`{=tex} (V^R(x_t)-`\hat `{=tex}V_t^R)\^2 \]

其中：

\[ `\hat `{=tex}V_t\^R \]

为基于 reward trajectory 计算的 value target。

Constraint Critic 对每项约束独立优化：

\[ L\^{C_j} = `\frac12`{=tex} (V^{C_j}(x_t)-`\hat `{=tex}V_t^{C_j})\^2
\]

其中：

\[ `\hat `{=tex}V_t\^{C_j} \]

为对应约束代价 trajectory 的 value target。

------------------------------------------------------------------------

2.6.6 拉格朗日乘子更新

对于第 (j) 项长期约束，定义 batch 内有效约束代价：

\[ `\hat `{=tex}J_j = `\frac{
\sum_t \chi_{j,t}v_{j,t}
}`{=tex} { `\sum`{=tex}*t`\chi`{=tex}*{j,t} } \]

仅当：

\[ `\sum`{=tex}*t`\chi`{=tex}*{j,t}\>0 \]

时更新：

\[ `\lambda`{=tex}\_j `\leftarrow`{=tex} `\max`{=tex} `\left`{=tex}( 0,
`\lambda`{=tex}*j+ `\eta`{=tex}*{`\lambda`{=tex}\_j}
(`\hat `{=tex}J_j-d_j) `\right`{=tex}) \]

其中：

  符号                                   定义
  -------------------------------------- -------------------------
  (`\hat `{=tex}J_j)                     当前 batch 平均约束代价
  (d_j)                                  约束预算
  (`\eta`{=tex}\_{`\lambda`{=tex}\_j})   拉格朗日学习率

若整个 batch 中该约束未生效，则：

\[ `\lambda`{=tex}\_j\^{new}=`\lambda`{=tex}\_j\^{old} \]

不得将未生效样本视为零违反样本。

------------------------------------------------------------------------

2.7 核心符号统一定义

  符号                   定义
  ---------------------- --------------------------
  \(N\)                  水库数量
  \(i\)                  水库索引
  \(t\)                  时间步
  (s_t)                  环境状态
  (x_t)                  集中式 Critic 输入状态
  (o_i\^{base})          基础观测
  (o_i\^{dec})           真实决策观测
  (a_i)                  第 (i) 个智能体动作
  (Q\^{tar})             动作映射得到的目标总下泄
  (Q\^{exec})            环境实际执行总下泄
  (V_i)                  库容
  (h_i)                  水位
  (I_i)                  实际总入流
  (I_i\^{int})           区间入流
  (r_t)                  系统奖励
  (v\_{j,t})             长期约束代价
  (A\^R)                 奖励优势
  (A\^{C_j})             约束优势
  (`\lambda`{=tex}\_j)   拉格朗日乘子

三、五库实例的数据与参数设置

本部分属于"乌东德---白鹤滩---溪洛渡---向家坝---三峡"研究实例的工程化设置，不属于专利对一般梯级系统的唯一限定。后续只要有真实数据或实验设计依据，可以修改本部分的数值，但修改后必须保证第二部分所述方法逻辑不变。

3.1 时间、拓扑与 CSV/数据字段

当前时间尺度为日尺度，(`\Delta`{=tex}t=86400) s，仿真区间为 2023-01-01
至 2025-12-31，不显式模拟相邻水库河道传播时滞。

数据源中至少需要明确区分最上游外部来水与各相邻水库之间的区间入流。当前字段语义为：

  字段                        含义
  --------------------------- -------------------------
  `WDD_external_inflow_m3s`   乌东德外部来水
  `BHT_interval_inflow_m3s`   乌东德---白鹤滩区间入流
  `XLD_interval_inflow_m3s`   白鹤滩---溪洛渡区间入流
  `XJB_interval_inflow_m3s`   溪洛渡---向家坝区间入流
  `THR_interval_inflow_m3s`   向家坝---三峡区间入流

下游 CSV 或其他数据源中的 `*_interval_inflow_m3s`
必须保持"区间入流"语义，不能在数据预处理阶段提前累加成总入流。下游当日真实总入流只能在
S4 中由"直接上游当日实际总下泄 + 当前区间入流"形成。S2
中由上游候选动作形成的 candidate inflow
只用于兼容性判断，禁止写回环境状态。

3.2 动作、观测与全局状态

当前五库离散动作数为

``` yaml
action:
  num_actions: 401
```

当前 `ReleaseActionMapper` 使用线性映射，即允许的严格单调
(`\psi`{=tex}\_i)
的一个具体实例。动作数不得在代码其他位置重复写死。每座水库各自的
`map_min_release_m3s` 和 `map_max_release_m3s`
应由配置或真实物理参数提供，不在本文无依据补值。

当前 base observation 维度为 17。乌东德的实际 Actor 输入维度为
17；其余四座下游水库在 base observation
后额外加入当前真实总入流，因此实际 decision observation 维度为 18：

``` text
WDD  = 17
BHT  = 18
XLD  = 18
XJB  = 18
THR  = 18
```

当前下游附加入流特征采用

``` text
encoded_current_inflow
=
current_inflow_m3s / map_max_release_m3s
```

这一归一化只作用于神经网络输入，不得改变真实 m³/s
入流、候选下泄量、安全判断或物理执行结果。

集中式 Critic 当前使用五座水库的 base observation 拼接形成 pre-action
centralized state，因此

\[ `\dim`{=tex}(x_t)=5`\times17`{=tex}=85. \]

当前 `share_observation` 维度为 85。HARL wrapper 若要求所有 agent 的
observation shape
一致，可以对接口层采用零填充或其他占位方式，但这种补值仅服务于张量形状，不能被解释为新的物理信息。训练时重新计算动作概率所用的
observation 仍必须对应各智能体真实采样时的 (o\^{dec}\_{i,t})。

Actor 当前不共享参数：

``` yaml
share_param: false
```

3.3 第一层安全约束的当前参数

P1 不允许通过"调参"主动放宽。P1
的具体物理边界必须来自真实水库参数、设备能力、Z-V
曲线以及合法运行边界。如果项目尚无可靠的紧急强制泄洪触发参数，不能自行创造。

P2
是各运行阶段的时变水位控制边界。本文保留其方法地位，但当前原文档没有提供一套可确认的完整
P2 数值表。因此各库防洪期、蓄水期、消落期等时变水位上下限以及 P2
最大允许松弛范围，必须从真实调度规则或配置文件中补充，并明确记录生效日期、边界值和最大允许松弛幅度。

P3 当前配置为：

``` yaml
safety:
  p3:
    release_change_ratio: 0.5
    release_change_floor_m3s: 5000.0
    max_relaxation_factor: 2.0
    level_change_limit_m:
      WDD: 0.30
      BHT: 0.50
      XLD: 0.90
      XJB: 0.20
      THR: 1.40
```

正常下泄变化约束为

\[ \|Q_t-Q\_{t-1}\| `\le`{=tex} `\max`{=tex} `\left`{=tex}( 0.5Q\_{t-1},
5000 `\right`{=tex}). \]

P3 松弛时对允许变化范围使用 relaxation
factor，但不能超过配置的最大值。首日若不存在可信的上一日实际下泄量，则设置

``` text
previous_release_m3s = None
```

此时只关闭确实依赖上一时刻下泄的 P3
子约束，不得伪造历史下泄量；如果当前水位是真实已知的，依赖当前水位与下一时刻水位的变化限制仍可以正常生效。

P4 当前形式为

\[ Q\^{tar}\_{i,t} `\ge`{=tex} `\eta`{=tex}*4 I*{i,t}. \]

当前配置为：

``` yaml
safety:
  p4:
    normal_release_inflow_ratio: 1.0
    min_release_inflow_ratio: 0.75
```

P4 在其适用运行阶段正常生效时，从 S201
开始就必须参与局部可行性和联合可行性判断；S3
只负责在联合可行集为空后逐步降低该要求，而不是等到空集出现后才首次启用
P4。

3.4 发电奖励与长期约束参数

当前发电奖励归一化尺度采用

\[ E\^{ref} = `\left`{=tex}( `\sum`{=tex}\_iP_i\^{inst}
`\right`{=tex})`\Delta`{=tex}t. \]

长期约束的以下参数必须逐项按"水库---约束类型"配置：

``` text
constraint_id
active rule / χ
normalization z_j
tolerance epsilon_tol_j
budget d_j
lambda_j
lambda_lr eta_lambda_j
```

同时还需要真实任务参数，例如生态下泄阈值 (Q_i\^{eco})、保证出力阈值
(P_i\^{gua})、航运下泄阈值 (Q_i\^{nav})
及其生效时段。这些参数如果当前数据或配置没有可靠依据，本文不填写假定数值。

3.5 HARL 当前运行约束

当前 S4 的顺流条件采样需要 runner
访问真实环境对象并调用准备阶段接口，因此现阶段工程设置为：

``` text
n_rollout_threads = 1
n_eval_rollout_threads = 1
ShareDummyVecEnv
```

在没有增加 RPC 或等价的可序列化 `prepare_step` /
顺序采样通信机制之前，不应直接切换到
`ShareSubprocVecEnv`，也不能为了并行化而改成五个 Actor 同时采样。

四、程序模块、数据契约与验证要求

4.1 程序模块职责

当前推荐的核心目录与职责如下。类名或文件名若与真实仓库已有接口不同，可以按真实代码调整，但职责边界不应混乱。

``` text
harl/envs/cascade_reservoir/
    action_mapping.py
    cascade_reservoir_env.py
    constraint_costs.py
    data_loader.py
    reservoir_physics.py
    routing.py
    sequential_sampler.py
    safety/
        constraint_context.py
        s201_local_feasibility.py
        s202_compatibility.py
        s203_extendability.py
        s3_relaxation.py

harl/runners/
    cascade_reservoir_runner.py
```

各模块职责如下：

  --------------------------------------------------------------------------------------------------
  模块                            主要职责
  ------------------------------- ------------------------------------------------------------------
  `data_loader.py`                读取、校验并提供外部来水、区间入流及运行参数，不进行隐式物理推演

  `reservoir_physics.py`          Z-V 关系、水量平衡、过机/弃水、功率、能量及状态转移

  `action_mapping.py`             离散动作索引与目标总下泄量之间的映射，不负责动态可行性判断

  `constraint_context.py`         保存当前决策步实际生效的 P1-P4 边界及 S3 松弛状态

  `s201_local_feasibility.py`     给定入流和约束上下文下的单库局部可行性

  `s202_compatibility.py`         构造相邻水库候选动作兼容矩阵

  `s203_extendability.py`         从三峡向乌东德逆向递推完整路径可延拓性并生成安全 mask

  `s3_relaxation.py`              P1 固定下执行 P4→P3→P2 分级最小松弛及最终最小违反 fallback

  `sequential_sampler.py`         WDD→THR 真实条件采样，并保存各 Actor 的真实行为条件

  `cascade_reservoir_env.py`      `reset`、`prepare_step`、`step`、物理执行、reward
                                  及长期约束原始违反量

  `cascade_reservoir_runner.py`   Buffer、Reward/Constraint Critics、GAE、HAPPO、Lagrangian
                                  及训练流程连接
  --------------------------------------------------------------------------------------------------

4.2 一次决策步的数据契约

`prepare_step` 读取当前物理状态和当日外部数据，构造全部 base
observations 和 pre-action centralized state (x_t)，运行正常
S2，必要时执行
S3，并输出当前决策步安全结构。该阶段不得推进环境物理状态。

`collect` 使用 `prepare_step` 的结果，严格按 WDD→BHT→XLD→XJB→THR
顺序采样。每确定一个上游动作后，计算直接下游真实入流、更新下游
(o\^{dec})、选取与已选上游动作匹配的 conditional
mask，再完成下游动作采样。

`env.step`
只接受最终完整联合动作，并输出下一状态、系统发电奖励、各长期约束原始违反量
(`\xi`{=tex})、生效标志 (`\chi`{=tex}) 以及诊断信息。

`insert` 必须保证同一时间索引内保存：

``` text
per-agent:
  o_dec[t]
  action_mask[t]
  action[t]
  old_log_prob[t]

shared:
  x_t
  reward[t]
  raw_violation[j,t]
  active_flag[j,t]
```

`compute` 构造 (v\_{j,t})，分别计算 Reward GAE、各 Constraint
GAE、价值目标和 Lagrangian Advantage。

`train` 更新 Reward Critic、各 Constraint Critic，随机生成 HAPPO agent
更新顺序，累计前序已更新 Agent 的最终 ratio，完成各 Actor 更新，并在
Actor/Critic 当前轮结束后更新各长期约束的 (`\lambda`{=tex}\_j)。

4.3 必须长期保持的程序不变量

水力因果关系必须始终为：

``` text
上游动作确定
→ 上游 Q_exec 确定
→ 下游真实总入流确定
→ 下游 o_dec 确定
→ 下游 conditional mask 确定
→ 下游 Actor 采样
```

除最终最小违反 fallback 外，真实执行的联合动作必须位于 S2/S3
已证明可完整延拓的路径内。P1 在任何 recovery 和 fallback
中都不能主动放宽。

当前假定 `Q_exec == Q_target`。若将来不再成立，必须重新审查 S202
候选兼容性与 S4 真实执行的一致性。

Critic 当前时点必须保持：

``` text
prepare_step 生成的 x_t
=
Reward Critic share_obs[t]
=
所有 Constraint Critic share_obs[t]
```

同一行为时间索引必须保持：

``` text
o_dec[t]
mask[t]
action[t]
old_log_prob[t]
```

四者严格同步。

若某长期约束在整个 batch 内没有有效生效样本，则对应 (`\lambda`{=tex}\_j)
保持不变。`prepare_step`、S201、S202、S203 和 S3
可行性搜索均不得提前推进库容、水位、上一时刻动作等环境物理状态。

4.4 训练前验证顺序

训练前应按完整数据链逐段验证，而不是仅检查程序是否能运行。

第一阶段检查 `reset → prepare_step`：base observation shape、centralized
state shape、S201/S202/S203 结果、S3 结果、root/conditional masks，以及
`prepare_step` 是否无物理状态副作用。

第二阶段检查 `prepare_step → collect`：WDD→THR
顺序是否真实发生、每个下游 `inflow_m3s`、每个 (o\^{dec})、每个
conditional mask、采样 action、old log-prob 和 RNN state 是否同步。

第三阶段检查 `collect → env.step`：`Q_exec`
与动作映射一致性、上下游实际入流一致性、水量平衡、Z-V
关系、P1、reward、(`\xi`{=tex}*{j,t}) 和 (`\chi`{=tex}*{j,t})。

第四阶段检查 `env.step → insert`：buffer
时间索引、`o_dec/mask/action/log_prob` 同步、(x_t) 对齐和 constraint
buffer 完整性。

第五阶段检查 `rollout → compute`：Reward GAE、各 Constraint
GAE、(v\_{j,t})、value targets 与 (A_t\^L)。

第六阶段检查 `compute → train`：Reward Critic、K 个 Constraint
Critic、固定的 lambda snapshot、随机 HAPPO agent order、前序已更新 Agent
最终 ratio 的累计、每个 Agent 更新后的 ratio 重算，以及 `active_count=0`
时 (`\lambda`{=tex}) 不变。

最后执行端到端 smoke test：

``` text
reset
→ prepare_step
→ collect
→ env.step
→ insert
→ compute
→ train
```

至少应覆盖正常可行、P4 需要松弛、P4 最大仍不足而需要 P3、P4/P3
最大仍不足而需要 P2、全部运行约束达到最大松弛仍需最终最小违反
fallback，以及某长期约束整个 batch 不生效等场景。

4.5 实验记录指标

第一层至少记录 P1 violation count、normal 比例、p4_relaxed
比例、p3_relaxed 比例、p2_relaxed 比例、min_violation_fallback
比例、各级实际松弛幅度、root safe action count
以及完整联合可行路径存在性。

水力一致性至少记录各库 water balance residual、下游实际总入流一致性、Z-V
合法范围和 (Q^{exec}-Q^{tar})。

第二层至少记录各项 (`\xi`{=tex}\_j)、(v_j)、active count、budget
(d_j)、(`\lambda`{=tex}\_j) 演化和各 Constraint Critic loss。

强化学习训练至少记录 system reward、total generation、Actor loss、Reward
Critic loss、Constraint Critic loss、entropy、agent-wise importance
ratio 和 cumulative (`\mu`{=tex})。

五、五库实例具体参数配置

本部分用于给出乌东德---白鹤滩---溪洛渡---向家坝---三峡五库串联系统在当前研究中的具体参数配置。与前文方法部分不同，本部分不定义新的算法流程，仅规定模型运行所需的实例化参数、数据来源以及参数修改边界。

【参数来源】

五库实例中的物理参数优先来源于实际数据文件。若程序默认值、配置文件参数或其他说明文档与数据文件存在冲突，应以数据文件中的值为准。

当前主要数据文件包括：

``` text
reservoir_physics.csv
water_level_limits.csv
level_storage_anchors.csv
initial_state.csv
```

其中：

-   `reservoir_physics.csv`：水库固定物理参数；
-   `water_level_limits.csv`：水位边界参数；
-   `level_storage_anchors.csv`：库容---水位关系；
-   `initial_state.csv`：仿真初始状态。

5.1 水库固定物理参数

水库固定物理参数由 `reservoir_physics.csv`
提供，包括装机容量、最大过机流量、最大总泄流能力以及代表净水头。

当前五库参数如下：

  ------------------------------------------------------------------------------------
  水库       装机容量/MW   最大过机流量/(m³/s)   最大总泄流能力/(m³/s)   代表净水头/m
  --------- ------------- --------------------- ----------------------- --------------
  WDD           10200             8500                   37362               135

  BHT           16000             8621                   42350               202

  XLD           12600             7749                   50000               197

  XJB           6400              7040                   48660               100

  THR           22500             31200                 100000                85
  ------------------------------------------------------------------------------------

上述参数用于发电计算、P1物理约束判断、S201局部可行性判断以及环境状态转移。

5.2 水位边界参数

水位边界由 `water_level_limits.csv` 提供。

  ------------------------------------------------------------------------------
  水库        最低安全水位/m   正常最高水位/m   防洪限制水位/m   安全最高水位/m
  ---------- ---------------- ---------------- ---------------- ----------------
  WDD              945              975              952             986.17

  BHT              765              825              785             832.34

  XLD              540              600              560             609.47

  XJB              370              380              370             383.00

  THR              145              175              145             180.40
  ------------------------------------------------------------------------------

其中最低安全水位和安全最高水位用于P1绝对安全边界；正常最高水位和防洪限制水位用于运行阶段约束。

P1不参与S3松弛。

5.3 库容---水位关系参数

库容---水位关系由 `level_storage_anchors.csv` 提供。

根据离散锚点建立：

\[ V_i=f_i(h) \]

以及：

\[ h_i=f_i\^{-1}(V) \]

用于水量平衡后的水位计算、S201判断以及水位约束检查。

当前版本采用真实锚点数据插值，不再使用人工设定的线性Z-V关系替代数据文件。

5.4 初始状态参数

初始状态由 `initial_state.csv` 提供。

当前仿真起始日期为：

``` text
2023-01-01
```

初始库容如下：

  水库    初始库容/m³
  ------ -------------
  WDD     5772830000
  BHT     18846620000
  XLD     11354940000
  XJB     4923950000
  THR     38689950000

环境初始化时直接读取：

\[ V\_{i,0}=V_i\^{initial} \]

并根据库容---水位关系计算初始水位。

5.5 动作映射参数

当前五库采用：

``` yaml
action:
  num_actions: 401
```

动作索引：

\[ k=0,`\ldots`{=tex},400 \]

归一化：

\[ u(k)=frac{k}{400} \]

当前采用线性动作映射：

\[ `\psi`{=tex}\_i(u)=u \]

目标总下泄量：

\[ Q\^{tar}\_{i,t} = Q\^{map,min}\_i+
(Q\^{map,max}\_i-Q\^{map,min}\_i)`\psi`{=tex}\_i(u) \]

其中动作映射范围不等同于动态安全范围。

5.6 安全约束参数

P1由真实物理参数、水位边界和库容---水位关系共同确定，包括：

-   水量平衡；
-   绝对水位边界；
-   最大泄流能力；
-   设备物理限制。

P2为运行阶段水位控制约束，其具体边界和松弛范围由调度规则或配置文件提供。

P3用于限制相邻时段运行变化。

P4采用：

\[ Q\^{tar}\_{i,t}`\geq`{=tex}`\eta`{=tex}*4 I*{i,t} \]

具体比例和生效时间由配置文件维护。

S3保持：

\[ P4ightarrow P3ightarrow P2 \]

的松弛顺序。

5.7 长期运行约束参数

长期运行约束属于S5，不替代第一层安全机制。

每个"水库---约束类型"独立配置：

``` text
constraint_id
active_rule
threshold
z_j
epsilon_tol_j
d_j
lambda_j
eta_lambda_j
```

包括生态下泄、保证出力和航运约束。

若：

\[ `\chi`{=tex}\_{j,t}=0 \]

则该约束不产生训练压力，并且对应拉格朗日乘子保持不更新。

参数更新、数据更新和方法结构修改必须严格区分。只有方法结构变化时，才需要重新检查S1→S2→S3→S4→S5完整逻辑链。

本文档的最终使用原则是：第二部分定义方法，第三部分定义五库实例如何赋值和读取数据，第四部分定义程序如何实现并验证，第五部分定义哪些地方目前不能自行补值。后续新增
CSV、配置项、水库参数或训练超参数时，应优先放入第三部分；新增代码接口、测试要求或实现限制时，应放入第四部分；只有改变算法本身时才允许修改第二部分，并且修改前必须重新核对专利的
S1→S2→S3→S4→S5 因果链；第五部分的参数不能随意修改。
