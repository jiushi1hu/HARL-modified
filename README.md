# 五库串联梯级水库双层安全多智能体强化学习

本项目基于 PKU-MARL/HARL，研究 WDD → BHT → XLD → XJB → THR 五库联合调度。
当前运行入口仅支持 `cascade_reservoir` 环境和 `happo` 算法。

## 启动训练

在项目根目录、已配置好的 `harl` 环境中运行：

```bash
conda activate harl
python -m examples.train --algo happo --env cascade_reservoir --exp_name cascade_happo_formal_seed1
```

也可以执行 `bash examples/train.sh --exp_name cascade_happo_formal_seed1`。
训练和环境参数分别来自 `harl/configs/algos_cfgs/happo.yaml`、
`harl/configs/envs_cfgs/cascade_reservoir.yaml`，命令行覆盖参数的方式保持不变。
`--load_config` 仍可加载已有水库实验配置；`model_dir` 和评估设置仍使用原配置接口。

运行依赖包括 PyTorch、NumPy、pandas、Gym、PyYAML、tensorboardX、TensorBoard、setproctitle。
当前本机验证环境为 Python 3.8、NumPy 1.24.4、PyTorch 2.4.1。
若旧 TensorBoard/protobuf 组合报兼容性错误，可在启动前设置：

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

## 保留的核心目录

- `harl/envs/cascade_reservoir/`：水库物理、数据加载、S201/S202/S203/S3/S4、环境和日志。
- `harl/runners/`：水库 Runner 及其必需的两个 on-policy 父类。
- `harl/algorithms/`：HAPPO、Reward/Constraint Critics、LagrangianManager。
- `harl/common/`、`harl/models/`、`harl/utils/`：训练依赖的公共组件。
- `data/cascade_reservoir/`：研究输入数据。
- `examples/`：训练入口与水库测试。
- `docs/`：研究方法和代码说明。

修改研究代码前阅读 `AGENTS.md` 和 `docs/论文与代码统一上下文_严谨修订版.md`。
严谨修订版与现有实现的差异需单独处理；本次目录清理没有改变算法语义。

## 按顺序验证

```bash
python -m examples.test_cascade_reservoir
python -m examples.test_cascade_reservoir_step3
python -m examples.test_cascade_training_integration
python -m unittest examples.test_cascade_training_order
```

前两项覆盖 Step 1～3，集成测试覆盖 Step 4～7 和模型保存/恢复。
测试使用临时输出目录；原训练结果、模型和日志保留。
历史清理记录已从 `docs/` 移除；当前规范和仍在使用的实验依据保留在 `docs/` 中。

## 上游来源与引用

HAPPO 及公共训练组件来源于 [PKU-MARL/HARL](https://github.com/PKU-MARL/HARL)。
原通用环境、其他算法及对应调优配置已从当前工作树移除，历史版本保存在 Git 中。
以下保留上游项目提供的论文引用：

```tex
@article{JMLR:v25:23-0488,
  author  = {Yifan Zhong and Jakub Grudzien Kuba and Xidong Feng and Siyi Hu and Jiaming Ji and Yaodong Yang},
  title   = {Heterogeneous-Agent Reinforcement Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {32},
  pages   = {1--67},
  url     = {http://jmlr.org/papers/v25/23-0488.html}
}
```

```tex
@inproceedings{
liu2024maximum,
title={Maximum Entropy Heterogeneous-Agent Reinforcement Learning},
author={Jiarong Liu and Yifan Zhong and Siyi Hu and Haobo Fu and QIANG FU and Xiaojun Chang and Yaodong Yang},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=tmqOhBC4a5}
}
```
