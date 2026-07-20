# Project Structure

> Last updated: 2026-07-20
> Keep this file in sync whenever directories or files are added, removed, or renamed.

---

## Top-level layout

```
battery-soc-transfer-benchmark/
│
├── src/battery_benchmark/   核心 Python 库（所有可复用代码）
├── configs/                 实验配置（YAML，改实验只改这里）
├── scripts/                 命令行入口脚本
├── notebooks/               Colab 笔记本（数据准备 + 实验编排）
├── tests/                   pytest 测试套件
├── docs/                    协议文档 + 审计报告
├── data/                    数据目录（不提交到 Git）
├── external_repos/          第三方参考代码（只读，不修改）
├── outputs/                 训练输出（不提交到 Git）
├── .venv/                   Python 虚拟环境（不提交到 Git）
├── pyproject.toml           包配置 + 依赖声明
├── requirements.txt         pip 依赖锁定
├── Makefile                 常用命令快捷方式
└── PROJECT_STRUCTURE.md     本文件
```

---

## `src/battery_benchmark/` — 核心库

### `data/` — 数据读取层

每个数据集对应一个适配器文件，负责读取原始文件并返回统一格式的 DataFrame。

| 文件 | 数据集 | 状态 |
|------|--------|------|
| `schema.py` | 统一列名定义（benchmark-level + adapter-level schema 常量） | ✅ |
| `base_dataset.py` | 所有适配器的抽象基类（DataFrame API） | ✅ |
| `registry.py` | 通过字符串名称查找数据集类 | ✅ |
| `synthetic.py` | 合成数据生成器（无需真实文件，用于 smoke test） | ✅ |
| `adapter_qc.py` | `QCResult` 容器 + adapter 自定义异常 | ✅ |
| `adapter_common.py` | manifest+mat 适配器公共逻辑（字段映射、QC、ManifestMatAdapter 基类） | ✅ |
| `sequence_record.py` | `SequenceRecord` dataclass — adapter 统一输出格式 | ✅ |
| `mat_loader.py` | 通用只读 MATLAB `.mat` 加载器（返回 `MatLoadResult`） | ✅ |
| `manifest_loader.py` | 通用 manifest CSV 加载器（返回 `list[ManifestEntry]`） | ✅ |
| `panasonic.py` | Panasonic NCR18650PF 适配器（`PanasonicAdapter`） | 🔶 第一版（读取 + 映射 + QC） |
| `lg_hg2.py` | LG HG2 适配器（`LGHG2Adapter`） | 🔶 第一版（读取 + 映射 + QC） |
| `samsung_30t.py` | Samsung INR21700-30T 适配器（`Samsung30TAdapter`） | 🔶 第一版（读取 + 映射 + QC） |
| `samsung_48g.py` | Samsung INR21700-48G 适配器 | ❌ 数据集未获取 |
| `cmu_evtol.py` | CMU eVTOL 适配器 | ⏳ 待实现 |
| `cmu_evtol_raw.py` | CMU KiltHub Version 3 资产清单、canonical raw ingestion、continuous-block QC | ✅ |
| `splits.py` | 数据集划分逻辑（按 cycle / cell / profile / temperature） | ✅ |
| `cycle_dataset.py` | 将 numpy 数组封装为 PyTorch Dataset | ✅ |
| `datamodule.py` | 组合 source + target DataLoader | ✅ |
| `collate.py` | 变长序列的 batch collate 函数 | ✅ |
| `vidal_sequence_v2_dataset.py` | 只读加载 sequence v2 index/Parquet，并执行 Case 1 QC | ✅ |
| `vidal_case1_windows.py` | record/segment-safe 窗口与 padding mask | ✅ |

**Panasonic / LG / Samsung Adapter 当前实现范围（`PanasonicAdapter` / `LGHG2Adapter` / `Samsung30TAdapter`）：**
- ✅ 读取各自 Vidal 2019 25 °C manifest CSV
- ✅ 按 split 选择文件（full_train / partial_train / test / exclude）
- ✅ 调用 `load_mat_file()` 读取原始 `.mat` 文件（只读）
- ✅ 字段映射：Time→time_s, Voltage→voltage_v, Current→current_a, Battery_Temp_degC→temperature_c, Ah→cycler_ah
- ✅ Adapter-level QC（共享 `adapter_common.py`）
- ✅ `metadata["split"]` = requested_split；`metadata["manifest_splits"]` 保存 manifest membership
- ✅ `paper_profile` 仅来自 manifest，不做 profile fallback
- ✅ `record_role` 来自 manifest（`model_input` / `soc_anchor` / `excluded`），Adapter 只读取并校验
- ✅ 严格 `soc_init` 解析（LG manifest 列）；selected drive-cycle SOC metadata 终态校验
- ✅ `summarize()` 校验 records 的 `metadata["split"]` 与请求 split 一致
- ❌ SOC 计算 / 清洗 / 重采样 / 归一化 / 切窗口（均未实现）

**训练输入筛选契约（文档约定，尚未实现过滤器）：**

Vidal-compatible 协议下，``record.metadata["record_role"] == "model_input"`` 的
selected drive-cycle **与** selected CC charge 记录均可进入标签与模型管线。
``file_type`` 决定 SOC 公式（``drive_discharge`` vs ``charge``）。
``excluded`` 记录不参与模型输入。

**Vidal-compatible SOC reconstruction（``labels/`` 模块）：**

- ``SOCCalculator.calculate_vidal()`` — v1 open-reference per-record Ah 归一化（``outputs/labelled/vidal_compatible_v1/``）
- ``scripts/generate_vidal_sequence_soc_v2.py`` — v2 sequence-aware 正式 benchmark label（``outputs/labelled/vidal_sequence_v2/``）
- v1 定位：`open_reference_per_record_v1`；v2 定位：sequence-aware 连续 SOC（见 ``docs/vidal_sequence_soc_v2.md``）
- ``label_source = "vidal_compatible_cycler_ah"``（v1，非 sequence-aware）
- SOC labels are benchmark reference labels derived from cycler current/Ah measurements and protocol constraints, not independent ground truth.
- ``calculate_cmu()`` — 占位，未实现

**Samsung 特有说明：**
- SOC metadata 直接来自 manifest 的 ``soc_init`` / ``soc_anchor_source``（与 LG 相同，无 Panasonic 式推导）
- manifest 含完整 SHA-256；真实 selected 文件 SHA 验证通过
- ``dataset_identity_status = STRONGLY_SUPPORTED``（非 CONFIRMED）
- 尚未计算 SOC / 清洗 / 重采样 / 归一化 / 切窗口

### `preprocessing/` — 预处理流水线

| 文件 | 作用 |
|------|------|
| `normalization.py` | MinMaxNormalizer / StandardNormalizer（在训练集上 fit，禁止 target 泄漏） |
| `sequence_builder.py` | 将时序 DataFrame 切成固定长度窗口（fixed / sliding / full-cycle 三种模式） |
| `filtering.py` | 低通滤波、异常值过滤 |
| `resampling.py` | 重采样到统一频率（原始数据 10 Hz） |
| `segmentation.py` | 在时间间断处（>300s）切分连续记录 |
| `quality_control.py` | 检查缺失值、时间戳不单调、重复行等 |
| `columns.py` | 列名常量和映射 |

### `labels/` — SOC 标签计算

> 原始数据集**不含 SOC 列**，需要从 Ah 字段库仑积分计算。

| 文件 | 作用 |
|------|------|
| `coulomb_counting.py` | 库仑积分：`SOC(t) = SOC_init + Ah(t) / Q_nominal` |
| `capacity_lookup.py` | 各电池的标称容量（Panasonic 2.9Ah，LG/Samsung 3.0Ah 等） |
| `dynamic_capacity.py` | 动态容量估计（用于 CMU eVTOL 代理标签） |
| `pseudo_labels.py` | 为无标签目标域生成伪 SOC 标签 |
| `soc_types.py` | SOC 来源枚举（cycler_reference / coulomb_counted / pseudo） |
| `label_quality.py` | 标签质量指标（边界违反率、单调性等） |

### `models/` — 模型定义

| 文件 | 模型 | 状态 |
|------|------|------|
| `base.py` | 所有模型的抽象基类 | ✅ |
| `registry.py` | 通过字符串名称查找模型类 | ✅ |
| `vidal_lstm.py` | Vidal et al. (2019) LSTM：单层，hidden=10，3特征输入 | ✅ 架构正确，待验证细节 |
| `lstm.py` | 通用可配置 LSTM | ✅ |
| `gru.py` | GRU | ✅ |
| `cnn_lstm.py` | CNN + LSTM | ✅ |
| `tcn.py` | Temporal Convolutional Network | ✅ |
| `transformer_encoder.py` | Transformer Encoder | ✅ |

**待添加：**
- `tatn.py` — TATN（2D-CNN + BiLSTM，Shen et al. IEEE TPEL 2023）

### `transfer/` — 迁移学习方法

| 文件 | 方法 | 状态 |
|------|------|------|
| `base.py` | 所有迁移方法的抽象基类 | ✅ |
| `source_only.py` | Source-only baseline（在源域训练，直接测目标域） | ✅ |
| `finetune.py` | Fine-tuning（用少量目标域标签微调） | ✅ |
| `scratch.py` | Scratch baseline（在目标域从头训练） | ✅ |
| `mmd.py` | Maximum Mean Discrepancy | ⏳ 占位符 |
| `coral.py` | Correlation Alignment | ⏳ 占位符 |
| `dann.py` | Domain-Adversarial Neural Network | ⏳ 占位符 |
| `pseudo_label.py` | 伪标签自训练 | ⏳ 占位符 |
| `source_free.py` | Source-free 域适应 | ⏳ 占位符 |

**待添加：**
- `tatn_transfer.py` — TATN 的两阶段迁移（pretrain + adversarial + MMD + 伪标签）

### `training/` — 训练引擎

| 文件 | 作用 |
|------|------|
| `trainer.py` | 主训练循环（epoch / batch / 验证） |
| `checkpointing.py` | 保存/恢复断点（支持 Colab 断线续训） |
| `early_stopping.py` | Early stopping 逻辑 |
| `callbacks.py` | 训练回调接口 |
| `loops.py` | 单 epoch 训练/验证循环 |
| `schedulers.py` | 学习率调度器封装 |
| `reproducibility.py` | 随机种子固定（torch / numpy / random） |

### `evaluation/` — 评估

| 文件 | 作用 |
|------|------|
| `metrics.py` | MAE / RMSE / MAX 误差 / R² |
| `evaluator.py` | 跑完整评估，返回指标字典 |
| `plots.py` | SOC 预测曲线、误差分布图 |
| `domain_metrics.py` | 跨域迁移指标 |
| `physical_consistency.py` | 物理一致性（SOC 边界违反、单调性） |
| `profile_metrics.py` | 按 drive cycle profile 分组的指标 |
| `aggregation.py` | 多次实验结果汇总 |

### `experiments/` — 实验编排

| 文件 | 作用 |
|------|------|
| `runner.py` | `ExperimentRunner`：读 config → 建数据集 → 建模型 → 迁移 → 训练 → 评估 → 保存，一条龙 |
| `vidal_case1.py` | Vidal Case 1 独立训练、mask-aware 评估、Drive 输出与图表 | ✅ |
| `vidal_case2.py` | Vidal Case 2：加载正式 Case 1 source 权重，target full_train fine-tune，target test 评估 | ✅ |
| `result_schema.py` | 实验结果的标准数据结构 |
| `comparison.py` | 多实验对比汇总 |

### `utils/` — 工具函数

| 文件 | 作用 |
|------|------|
| `config.py` | 加载/合并 YAML 配置 |
| `paths.py` | 路径管理（本地 / Colab 两套） |
| `device.py` | 自动选 CPU / GPU |
| `logging.py` | 日志配置 |
| `environment.py` | 环境检测（Colab / 本地） |
| `git_info.py` | 记录当前 commit hash 到实验结果 |

---

## `configs/` — 实验配置

**改实验只需改这里的 YAML，不需要动 Python 代码。**

```
configs/
├── experiments/          完整实验配置（数据集 + 模型 + 迁移方法）
│   ├── smoke_test.yaml       合成数据快速验证（无需真实数据）
│   ├── vidal_case1.yaml      Panasonic → Panasonic（同域 baseline）
│   ├── vidal_case2.yaml      Vidal Case 2 formal defaults（source/target 由 CLI 指定）
│   ├── vidal_case2_smoke.yaml Vidal Case 2 smoke defaults（2 epochs）
│   ├── vidal_case2_diagnostic.yaml Vidal Case 2 5-epoch full-window development diagnostic
│   ├── vidal_case3.yaml      Panasonic → Samsung 48G（❌ 数据缺失）
│   ├── vidal_case4.yaml      Panasonic → Samsung 30T
│   ├── unified_lstm_finetune.yaml   统一 benchmark：fine-tuning
│   ├── unified_lstm_mmd.yaml        统一 benchmark：MMD
│   └── cmu_evtol_uda.yaml    CMU eVTOL 无监督域适应
│
├── datasets/             数据集元数据（容量、profiles、温度范围等）
│   ├── panasonic_18650pf.yaml
│   ├── lg_hg2.yaml
│   ├── samsung_30t.yaml
│   ├── samsung_48g.yaml
│   ├── cmu_evtol.yaml
│   └── synthetic.yaml
│
├── models/               模型超参数
│   ├── vidal_lstm.yaml
│   ├── lstm.yaml
│   ├── gru.yaml
│   ├── cnn_lstm.yaml
│   ├── tcn.yaml
│   └── transformer_encoder.yaml
│
├── transfer/             迁移方法参数
│   ├── source_only.yaml
│   ├── finetune.yaml
│   ├── scratch.yaml
│   ├── mmd.yaml
│   ├── coral.yaml
│   ├── dann.yaml
│   ├── pseudo_label.yaml
│   └── source_free.yaml
│
└── paths/                路径配置（本地 vs Colab）
    ├── local.yaml            本地运行：data/raw、outputs/ 等
    └── colab_drive.yaml      Google Drive 挂载路径
```

---

## `scripts/` — 命令行入口

| 脚本 | 用法 |
|------|------|
| `train.py` | 主训练入口：`python scripts/train.py --config ... --paths ...` |
| `evaluate.py` | 单独评估已有 checkpoint |
| `run_experiment.py` | 更完整的实验流程（含预处理检查） |
| `prepare_dataset.py` | 原始数据 → `data/processed/` 标准格式 |
| `validate_dataset.py` | 检查 processed 数据格式是否合规 |
| `aggregate_results.py` | 汇总多个实验的 metrics.json 到对比表 |
| `run_vidal_reproduction.py` | 批量跑 vidal_case1 ~ case4 |
| `run_vidal_case2.py` | Vidal Case 2 source→target transfer CLI（单方向或六方向 all-pairs） |
| `run_cmu_transfer.py` | CMU eVTOL 迁移实验 |
| `inventory_cmu_evtol_raw.py` | CMU Version 3 主时序/impedance 资产清单与 provenance |
| `qc_cmu_evtol_raw.py` | CMU Version 3 顺序式 per-cell continuous-block QC 汇总 |
| `audit_cmu_evtol_phase_evidence.py` | CMU 代表性 cells 的 mission/phase reconstruction 描述性证据审计（不生成正式标签） |
| `register_external_repo.py` | clone 外部仓库并记录到 registry |
| `audit_external_repo.py` | 记录外部仓库的 commit hash 和 license |
| `inspect_panasonic_25c_split.py` | 只读检查 Panasonic 25°C 文件，生成 manifest CSV 和 QC 报告 |
| `inspect_lg_hg2_25c_split.py` | 只读检查 LG HG2 25°C 文件，生成 manifest CSV 和 QC 报告（含 HWFET 数据质量故障检测） |
| `inspect_samsung_30t_25c_split.py` | 只读检查 Samsung 30T 25°C 文件，生成 manifest CSV；含 SHA-256、信号指纹、身份验证（T30 vs 30T）、20项验证及重复运行哈希稳定性 |

---

## `notebooks/` — Colab 笔记本

```
notebooks/
├── 00_colab_setup.ipynb          Colab 环境初始化、挂载 Drive、安装依赖
├── 01_dataset_inventory.ipynb    数据集清单检查
├── 02_prepare_panasonic.ipynb    Panasonic 数据预处理
├── 03_prepare_lg_hg2.ipynb       LG HG2 数据预处理
├── 04_prepare_samsung.ipynb      Samsung 30T / 48G 数据预处理
├── 05_prepare_cmu_evtol.ipynb    CMU eVTOL 数据预处理
├── 10_vidal_case1_baseline.ipynb Vidal Case 1：同域 baseline
├── 11_vidal_case2_finetuning.ipynb Vidal Case 2：跨化学体系 fine-tuning
├── 12_vidal_cases3_4.ipynb       Vidal Case 3 & 4：Samsung 目标域
├── 20_unified_backbone_baselines.ipynb 统一 backbone 各 baseline 对比
├── 30_unsupervised_domain_adaptation.ipynb MMD / CORAL / DANN
├── 31_source_free_adaptation.ipynb Source-free 方法
├── 40_cmu_evtol_transfer.ipynb   CMU eVTOL 迁移实验
├── 50_results_analysis.ipynb     全局结果分析与可视化
└── templates/
    ├── experiment_template.ipynb   新实验模板
    └── reproduction_template.ipynb 论文复现模板
```

---

## `tests/` — 测试套件

| 文件 | 覆盖内容 |
|------|---------|
| `test_smoke_training.py` | 端到端完整流程（合成数据） |
| `test_sequence_builder.py` | 时间窗口切片逻辑 |
| `test_normalization.py` | MinMax / Standard 归一化 |
| `test_splits.py` | 数据划分（无泄漏） |
| `test_schema.py` | 统一列名格式 |
| `test_metrics.py` | MAE / RMSE / MAX 计算 |
| `test_coulomb_counting.py` | 库仑积分 SOC 计算 |
| `test_checkpoint_resume.py` | 断点保存与恢复 |
| `test_cmu_evtol_raw.py` | CMU Version 3 schema、provenance、continuous-block 与真实资产 integration |
| `test_cmu_evtol_phase_evidence.py` | CMU 功率符号、block-safe Ns runs、真实时间持续时长与证据审计 integration |

运行：`.venv/bin/pytest tests/ -v`

---

## `docs/` — 文档

| 文件 | 内容 |
|------|------|
| `audit_report.md` | 数据集审计报告（599个 .mat 文件全检，含质量问题清单） |
| `dataset_inventory.csv` | 每个文件的统计信息（行数、采样率、电压/电流范围等） |
| `file_schema_summary.csv` | 各数据集的列名、dtype、电流符号规约 |
| `data_quality_issues.csv` | 282条数据质量问题（含严重程度和处理建议） |
| `paper_split_manifest.csv` | 每个文件的 train/test/partial_target 分类（审计参考，非最终 ground truth） |
| `panasonic_vidal2019_25c_split_report.md` | **Panasonic 25°C 文件划分报告**（Vidal 2019 复现；含 Cycle→Mix 映射验证、充电配对、QC） |
| `lg_hg2_vidal2019_25c_split_report.md` | **LG HG2 25°C 文件划分报告**（Vidal 2019 复现；含 Mixed→Mix 映射、HWFET 故障记录、充电配对、QC） |
| `samsung_30t_vidal2019_25c_split_report.md` | **Samsung 30T 25°C 文件划分报告**（Vidal 2019 复现；含 T30 vs 30T 身份审计、缺失 Cap_1C、HWFET 有效性确认、SHA-256、信号指纹） |
| `architecture.md` | 系统架构说明 |
| `benchmark_protocol.md` | 公平对比协议 |
| `dataset_protocol.md` | 数据集适配器规范（统一 schema、防泄漏要求） |
| `fairness_protocol.md` | Paper-faithful vs. unified benchmark 两种模式说明 |
| `label_protocol.md` | SOC 标签计算协议 |
| `reproduction_protocol.md` | Vidal 复现步骤 |
| `known_uncertainties.md` | 已知不确定项（Samsung 30T 命名已确认；48G 缺失） |
| `cmu_evtol_protocol.md` | CMU eVTOL 代理标签评估协议 |
| `colab_workflow.md` | Google Colab 使用流程 |
| `protocols/cmu_evtol_raw_ingestion_v0_1.md` | CMU Version 3 raw ingestion 与 continuous-block QC 规则 |
| `protocols/cmu_evtol_transfer_benchmark_v0_1.md` | CMU eVTOL transfer benchmark v0.1 cell-level 协议 |
| `audits/cmu_evtol_mission_phase_evidence_v0_1.md` | 代表性 cells 的 mission/phase reconstruction 证据审计与 PARTIAL 结论 |

---

## `data/` — 数据目录（不提交到 Git）

```
data/
├── raw/                      原始文件（不修改）
│   ├── Panasonic_18650PF/    ✅ 已获取，25degC + 多温度
│   ├── LG_HG2/               ✅ 已获取，-20~40degC
│   ├── Samsung_30T/          ✅ 已获取，-20~40degC（论文称"T30"，同一数据集）
│   ├── Samsung_48G/          ❌ 未获取，仅有 README 占位
│   ├── cmu_evtol_vtc6/
│   │   └── figshare_14226830/ CMU KiltHub eVTOL Battery Dataset Version 3（本地 raw，不提交）
│   └── LG 18650HG2 .../      参考用（含 MATLAB 脚本和已处理版本）
├── interim/                  中间处理产物
├── processed/                标准化 DataFrame，模型直接读取
└── manifests/                索引文件（可提交 Git）
    ├── panasonic_vidal2019_25c_manifest.csv      Panasonic 25°C 文件划分清单（61行，含 QC 元数据）
    ├── lg_hg2_vidal2019_25c_manifest.csv         LG HG2 25°C 文件划分清单（36行，含 HWFET FAIL 标记）
    ├── samsung_30t_vidal2019_25c_manifest.csv    Samsung 30T 25°C 文件划分清单（34行，含 SHA-256、信号指纹、身份状态）
    ├── cmu_evtol_vtc6_cell_protocols.csv         CMU 22-cell protocol/condition registry
    ├── cmu_evtol_vtc6_asset_inventory_v3.csv     CMU Version 3 主文件与 impedance provenance
    ├── cmu_evtol_vtc6_raw_qc_v3_120s.csv         CMU 22-cell exploratory 120 s raw block QC
    └── cmu_evtol_mission_phase_evidence_v0_1.csv CMU 代表性 cells 的描述性 phase/mission evidence
```

---

## `external_repos/` — 外部参考代码（不提交到 Git）

```
external_repos/
├── references/               只读参考（看架构/代码风格）
├── reproductions/            正在复现/对比的仓库
│   └── xf68bwh54v-1/         Vidal et al. 原始 MATLAB 代码
│       ├── LSTMvsFNN_Script_Nov_2021_V2.m   主训练脚本
│       └── Panasonic_NCR18650PF_Data_Normalized/  已归一化数据
├── archived/                 已完成检查，不再活跃
├── README.md                 使用规则（禁止在 src/ 中 import）
├── repo_registry.example.yaml
└── repo_registry.local.yaml  本地仓库记录（gitignored）
```

**规则：** `external_repos/` 里的代码只能看，不能直接用。需要迁移的代码必须重写、测试、注明出处后放入 `src/`。

---

## 数据流

```
data/raw/*.mat
    ↓  src/data/panasonic.py 等适配器
    读取 → 统一列名 → 库仑积分算 SOC → DataFrame
    ↓  scripts/prepare_dataset.py
data/processed/*.parquet
    ↓  src/preprocessing/normalization.py + sequence_builder.py
    归一化（训练集 fit）→ 切窗口 → (N, seq_len, 3) tensor
    ↓  src/training/trainer.py
模型训练
    ↓  src/evaluation/evaluator.py
outputs/<experiment_name>/
    ├── metrics.json
    ├── predictions.csv
    └── checkpoints/last.pt
```

---

## 当前实现状态

| 组件 | 状态 |
|------|------|
| 合成数据集 + smoke test | ✅ 完整 |
| 测试套件（69 个测试） | ✅ 全部通过 |
| VidalLSTM 架构 | ✅ 搭好，细节待核对 |
| Source-only / Fine-tuning / Scratch | ✅ 完整 |
| Trainer + 断点续训 | ✅ 完整 |
| 评估指标（MAE/RMSE/MAX） | ✅ 完整 |
| **Panasonic 适配器** | ⏳ **下一步** |
| **LG HG2 适配器** | ⏳ 待实现 |
| Samsung 30T 适配器 | 🔶 第一版（读取 + SHA + QC） |
| Samsung 48G 适配器 | ❌ 数据缺失 |
| MMD / CORAL / DANN | ⏳ 占位符 |
| TATN 模型 + 迁移方法 | ⏳ 待移植 |
| CMU eVTOL | ⏳ 待实现 |
