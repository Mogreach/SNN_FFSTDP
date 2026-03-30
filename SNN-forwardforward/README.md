# SNN-forwardforward

## 项目概述
本目录主要用于 `Forward-Forward + SNN` 方向的实验、调参、分析与部分硬件/能耗相关辅助脚本。

当前主训练入口以 `ff-snn.py` 为准，默认工作流是：

- 数据集加载
- 构建 `MLP` 或 `CNN` 网络
- 执行无监督训练
- 记录训练/验证/测试结果
- 导出曲线图、checkpoint 与 `metrics.json`

## 本次重构要点
本次重构目标是在**不改变现有默认逻辑**的前提下，把“实验模式切换”“指标采集”“训练编排”从网络细节里剥离出来，方便后续继续扩展。

### 1. 顶层实验模式统一
原先无监督模式下，手写梯度与 `loss.backward()` 自动微分的逻辑分散在网络代码和注释中，不利于切换和管理。

现在统一抽象为：

- `learning_mode`
  - 当前支持 `unsupervised`
  - 预留 `supervised` 扩展入口
- `unsupervised_update_mode`
  - `autograd`
  - `manual`

相关配置入口：

- `config.py`
- `ff-snn_hpo.py`

这样做之后，实验模式不再依赖源码内注释开关，而是在顶层统一设置。

### 2. 训练编排与网络实现解耦
原先 `ff-snn.py` 同时承担了：

- 数据集构建
- DataLoader 构建
- 模型构建
- 训练循环
- 验证/测试
- 指标统计
- 绘图与落盘

现在训练编排已集中到：

- `src/experiment_runner.py`

入口脚本 `ff-snn.py` 只保留参数解析和调用 runner 的职责，便于维护和后续扩展 supervised 分支。

### 3. 指标采集统一收口
原先指标记录逻辑大量散落在 `ff-snn.py` 中，包括：

- 每层 loss / goodness / cosine similarity / firing rate
- train accuracy
- GPU memory
- 手写梯度时间、峰值显存、算子量估计
- autograd comparison 的显存/时间

现在统一由：

- `src/metrics_tracker.py`

负责记录、聚合、绘图和导出，训练脚本只负责把每一步结果交给 tracker。

### 4. 无监督训练分支显式化
原先 `MLP` 无监督训练中，手写梯度、自动微分、对比统计、真实更新路径混在同一个大函数中，可读性较差。

现在 `src/ff_snn_mlp.py` 中已显式拆分为：

- 时序前向统计
- 手写梯度计算
- autograd 对比分支
- manual 更新
- autograd 更新

这样可以明确区分：

- 哪一部分只是为了统计和比较
- 哪一部分才是实际训练更新路径

### 5. 为 supervised 扩展预留结构
本次没有主动实现 supervised 训练逻辑，但已经把扩展点留在以下位置：

- `src/experiment.py`
- `src/experiment_runner.py`

后续如果增加 supervised 分支，原则上只需要：

1. 在 `ExperimentModeConfig` 中沿用现有模式配置
2. 在 runner 中补充 supervised 训练入口
3. 让网络返回同一类 step result / metrics 接口

即可尽量复用当前框架。

## 当前目录结构
以下为当前目录中主要内容的职责说明：

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
│  ├─ experiment.py
│  ├─ experiment_runner.py
│  ├─ metrics_tracker.py
│  ├─ ff_snn_mlp.py
│  ├─ ff_snn_cnn.py
│  ├─ generate_neg_sample.py
│  ├─ loss.py
│  └─ dataset.py
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
  - 只负责解析参数并调用 `run_experiment`

- `ff-snn_hpo.py`
  - 超参数搜索入口
  - 通过顶部常量设置实验模式、搜索空间和输出 CSV

- `config.py`
  - 统一实验参数解析
  - 当前已包含模式相关参数：
    - `learning_mode`
    - `unsupervised_update_mode`
    - `capture_manual_grad_metrics`
    - `capture_autograd_comparison`

### 训练与模式组织
- `src/experiment.py`
  - 定义实验模式对象与 step result 数据结构
  - 统一描述：
    - 当前跑的是哪种学习模式
    - 无监督下使用哪种更新方式
    - 当前 step 返回哪些统计结果

- `src/experiment_runner.py`
  - 负责训练 orchestration
  - 包含：
    - 数据集构建
    - DataLoader 构建
    - 模型构建
    - train / val / test 调度
    - 输出目录管理
    - metrics 落盘

- `src/metrics_tracker.py`
  - 统一指标收集与导出
  - 当前负责：
    - 每层曲线缓存
    - epoch 统计汇总
    - GPU memory 聚合
    - manual/autograd profiling 聚合
    - 训练曲线绘图
    - `metrics.json` 内容组织

### 网络实现
- `src/ff_snn_mlp.py`
  - 当前 `MLP` 主网络实现
  - 已完成本次重构中最主要的结构整理
  - 重点包括：
    - `Net`
      - 组织各层
      - 汇总每层 profiling 结果
      - 输出统一的 `UnsupervisedStepResult`
    - `Layer`
      - hidden layer 的无监督训练逻辑
      - 区分 manual / autograd 两种更新路径
      - 保留手写梯度与 autograd 对比能力
    - `OutputLayer`
      - 作为最后分类读出层
      - 当前仍使用交叉熵读出训练

- `src/ff_snn_cnn.py`
  - `CNN` 版本网络实现
  - 当前已对齐到与 MLP 相近的 step result 返回形式
  - 但整体结构整理程度还不如 `ff_snn_mlp.py`
  - 后续如果继续维护，建议按 MLP 的组织方式进一步收敛

### 训练辅助模块
- `src/generate_neg_sample.py`
  - 正负样本构造逻辑
  - 当前无监督训练 heavily 依赖这里生成正样本和负样本

- `src/loss.py`
  - 手写梯度、delta loss 等核心公式实现
  - 目前仍保留原始数学逻辑，未做公式层面的改写

- `src/dataset.py`
  - 数据集相关辅助逻辑

## 当前默认实验逻辑
当前默认配置下的主流程可以概括为：

- `learning_mode = unsupervised`
- `unsupervised_update_mode = autograd`
- hidden layer 按无监督 goodness 目标训练
- output layer 作为 readout / classifier 单独训练
- profiling 默认开启：
  - manual gradient metrics
  - autograd comparison

也就是说：

- **默认训练更新路径**仍是自动微分版本
- **手写梯度路径**现在可以通过顶层显式切换启用
- 即使不启用 manual 更新，也可以保留手写梯度统计/对比能力

## 输出与实验记录
当前实验输出通常位于：

- `logs/<learning_mode>/<dataset>/<model>/<run_name>/...`

其中会包含：

- `args.txt`
- `output_log.txt`
- `metrics.json`
- `checkpoint_max.pth`
- `checkpoint_last.pth`
- 各类训练曲线图

## 后续维护建议

### 推荐继续保持的原则
- 模式切换只在顶层配置做，不再依赖源码注释切换
- 指标采集统一走 tracker，不把统计逻辑重新塞回训练循环
- 网络文件内部优先拆成“小函数 + 明确职责”
- supervised 扩展优先复用现有 `ExperimentModeConfig` 与 runner 框架

### 后续可继续整理的点
- 将 `ff_snn_cnn.py` 进一步按 `ff_snn_mlp.py` 的结构风格收敛
- 为 supervised 分支补齐统一 step result
- 将一部分历史分析脚本与当前主实验入口做更明确区分
- 为 README 再补充典型运行命令示例

## 说明
本 README 重点记录的是**当前主干训练框架的结构与本次重构意图**，便于后续继续迭代，而不是覆盖目录中所有历史脚本的细节。
