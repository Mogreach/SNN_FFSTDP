# SNN-forwardforward

## 项目概述
本目录用于 `Forward-Forward + SNN` 方向的训练、调参、分析与部分硬件/能耗相关辅助脚本。

当前主训练入口是 `ff-snn.py`，统一工作流为：

- 数据集加载
- 根据 `model` 与 `learning_mode` 构建网络
- 执行训练 / 验证 / 测试
- 记录指标、导出曲线、保存 checkpoint

## 当前主干设计

### 1. 实验模式统一
当前实验模式由 `src/experiment.py` 中的 `ExperimentModeConfig` 统一描述，核心只有两组参数：

- `learning_mode`
  - `unsupervised`
  - `supervised`
- `hidden_layer_update_mode`
  - `autograd`
  - `manual`

这两组参数同时适用于 `MLP` 和 `CNN`，也同时适用于监督与无监督分支。

### 2. runner 统一调度
训练编排已经集中到 `src/experiment_runner.py`，`ff-snn.py` 只负责解析参数并调用 runner。

runner 当前会按如下方式分发模型实现：

- `MLP + unsupervised` -> `src/ff_snn_mlp_unsup.py`
- `MLP + supervised` -> `src/ff_snn_mlp_sup.py`
- `CNN + unsupervised` -> `src/ff_snn_cnn_unsup.py`
- `CNN + supervised` -> `src/ff_snn_cnn_sup.py`

### 3. 指标采集统一收口
训练过程中的 goodness、cosine similarity、spike rate、GPU memory、manual gradient profiling、autograd comparison 等指标，统一由：

- `src/metrics_tracker.py`

负责聚合、绘图和导出。

### 4. loss 定义集中管理
和 FF / delta-loss 相关的公式已经统一收口到：

- `src/loss.py`

这样后续如果需要修改或扩展不同 loss，不需要再分别改多个网络文件。

## 当前目录结构

```text
SNN-forwardforward/
├─ README.md
├─ config.py
├─ ff-snn.py
├─ ff-snn_hpo.py
├─ ff-snn_opt.py
├─ dataset_generate.py
├─ visualization_debug.py
├─ log.txt
├─ src/
│  ├─ __init__.py
│  ├─ dataset.py
│  ├─ experiment.py
│  ├─ experiment_runner.py
│  ├─ metrics_tracker.py
│  ├─ generate_neg_sample.py
│  ├─ loss.py
│  ├─ ff_snn_mlp.py
│  ├─ ff_snn_mlp_unsup.py
│  ├─ ff_snn_mlp_sup.py
│  ├─ ff_snn_cnn.py
│  ├─ ff_snn_cnn_unsup.py
│  └─ ff_snn_cnn_sup.py
├─ train_script/
├─ utils/
├─ ANN-FF_code/
├─ energy_cost/
├─ hardware_sim/
├─ data/
├─ logs/
├─ doc/
└─ images/
```

## 重要代码说明

### 顶层入口
- `ff-snn.py`
  - 当前主训练入口
  - 负责解析参数并调用 `run_experiment`

- `ff-snn_hpo.py`
  - 超参数搜索入口
  - 通过顶部常量设置 `MODEL`、`LEARNING_MODE`、`HIDDEN_LAYER_UPDATE_MODE` 和搜索空间

- `config.py`
  - 统一实验参数解析
  - 当前和模式相关的正式参数为：
    - `learning_mode`
    - `hidden_layer_update_mode`
    - `capture_manual_grad_metrics`
    - `capture_autograd_comparison`

### 训练与模式组织
- `src/experiment.py`
  - 定义 `ExperimentModeConfig`、`StepResult`、profiling 数据结构

- `src/experiment_runner.py`
  - 统一负责：
    - 数据集构建
    - DataLoader 构建
    - 模型构建
    - 按 `model + learning_mode` 分发训练入口
    - 评估与日志落盘

- `src/metrics_tracker.py`
  - 统一负责：
    - 每层指标缓存
    - epoch 汇总
    - GPU memory 聚合
    - manual/autograd profiling 聚合
    - 曲线绘图
    - `metrics.json` 导出

### 网络实现
- `src/ff_snn_mlp_unsup.py`
  - 当前无监督 MLP 主实现
  - hidden layer 支持 `manual / autograd`
  - 推理使用 `predict_multiple`
  - 通过标签假设逐类构造输入，比较 hidden goodness 后选最大值

- `src/ff_snn_mlp_sup.py`
  - 当前监督 MLP 主实现
  - hidden layer 支持 `manual / autograd`
  - output layer 负责最终分类读出

- `src/ff_snn_cnn_unsup.py`
  - 当前无监督 CNN 主实现
  - 结构上已尽量对齐 `ff_snn_mlp_unsup.py`
  - hidden layer 支持 `manual / autograd`
  - 推理同样使用 `predict_multiple`
  - 负样本生成与推理标签嵌入使用 `embed_label_onehot`

- `src/ff_snn_cnn_sup.py`
  - 当前监督 CNN 主实现
  - 结构上已对齐 `ff_snn_mlp_sup.py`
  - hidden layer 支持 `manual / autograd`
  - output layer 负责最终分类读出

### 兼容 / 历史文件
- `src/ff_snn_mlp.py`
  - 兼容层
  - 主要用于复用旧 import 路径，当前 re-export `ff_snn_mlp_unsup.py` 中的实现

- `src/ff_snn_cnn.py`
  - 历史 CNN 实验文件
  - 当前不是 `experiment_runner.py` 默认构建入口
  - 更适合作为旧逻辑参考，而不是继续扩展的新主干

### 训练辅助模块
- `src/generate_neg_sample.py`
  - 负责正负样本构造
  - 当前无监督 MLP / CNN 都依赖这里生成训练样本

- `src/loss.py`
  - 集中定义 FF goodness loss、supervised delta loss 以及手动梯度相关公式

- `src/dataset.py`
  - 数据集相关辅助逻辑

## 当前默认实验逻辑
当前默认配置为：

- `learning_mode = unsupervised`
- `hidden_layer_update_mode = autograd`
- `capture_manual_grad_metrics = True`
- `capture_autograd_comparison = True`

这意味着：

- 实际 hidden layer 更新默认走自动微分
- manual gradient 路径仍可通过顶层参数切换启用
- profiling 默认会保留 manual/autograd 对比统计

## 典型运行方式

### 直接训练
```bash
python ff-snn.py -model MLP -learning_mode unsupervised -hidden_layer_update_mode autograd
python ff-snn.py -model MLP -learning_mode supervised -hidden_layer_update_mode manual
python ff-snn.py -model CNN -learning_mode unsupervised -hidden_layer_update_mode manual
python ff-snn.py -model CNN -learning_mode supervised -hidden_layer_update_mode autograd
```

### HPO
`ff-snn_hpo.py` 中当前通过顶部常量控制实验：

- `MODEL`
- `LEARNING_MODE`
- `HIDDEN_LAYER_UPDATE_MODE`

修改后直接运行：

```bash
python ff-snn_hpo.py
```

## 输出与实验记录
当前实验输出通常位于：

- `logs/<learning_mode>/<dataset>/<model>/<run_name>/...`

其中常见内容包括：

- `args.txt`
- `output_log.txt`
- `metrics.json`
- `checkpoint_max.pth`
- `checkpoint_last.pth`
- 各类训练曲线图

## 当前维护建议

### 推荐继续保持的原则
- 模式切换只通过顶层参数控制，不再依赖源码注释切换
- 指标采集统一走 `metrics_tracker.py`
- 新的 loss 公式优先收口到 `loss.py`
- 新网络分支尽量保持 `Net / Layer / OutputLayer` 的清晰结构

### 后续仍可继续整理的点
- 将 `src/ff_snn_cnn.py` 进一步归档或显式标注为 legacy
- 为 `README` 补充更完整的数据集准备说明
- 继续减少历史脚本与当前主训练框架之间的歧义

## 说明
本 README 重点记录的是当前主干训练框架与实际接线方式，便于后续维护和扩展，不覆盖目录中所有历史实验脚本的细节。
