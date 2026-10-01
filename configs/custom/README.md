# 自定义配置说明

## 入口配置

- `FalconDet_train.yml`
  - 主训练入口；
  - 组合其他基础配置；
  - 只保存当前实验特有的覆盖项。
- `FalconDet_eval.yml`
  - 完整验证入口；
  - 不限制验证 batch 数量。
- `FalconDet_debug.yml`
  - 使用 AstroDim_mini 的快速连通性检查入口。

## 基础配置

- `AstroDim_dataset.yml`
  - 数据集路径、数据变换和 DataLoader。
- `FalconDet_model.yml`
  - 任务类型、共享模型参数、模型结构、loss 和后处理。
- `training.yml`
  - 训练周期、模块冻结、梯度裁剪、优化器、学习率调度和 EMA。
- `runtime.yml`
  - 日志、输出目录、checkpoint、评估和分布式运行参数。

## 配置覆盖原则

每个默认参数应尽量只在一个基础配置中定义。

入口配置只覆盖当前运行确实不同的参数，例如：

- `output_dir`
- debug batch 限制
- 临时数据集路径
- 当前实验的 epoch 或优化参数

不要在多个基础配置中重复定义同一个默认参数，否则配置合并后的实际来源会难以追踪。
