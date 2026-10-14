# OPFData 使用指南（下载位置 / 内容 / 用法）

OPFData 是 [arXiv:2406.07234](https://arxiv.org/abs/2406.07234) 发布的 AC-OPF 数据集，由 pglib-opf
算例生成。GridSFM 用它做 **微调** 和 **精度评测**（`model/examples/opfdata.py`）。
本文所有目录结构、字段和尺寸都是从本机实测得到的，不是转述文档。

---

## 1. 下载在哪里

数据不是 GridSFM 下的，而是由 PyG 的 `torch_geometric.datasets.OPFDataset` 下载到 **`root` 参数**指向的目录。
GridSFM 里三处控制这个 `root`：

| 位置 | `root` 取值 |
|---|---|
| `model/examples/opfdata.py:36` | `--root`，默认 `Path.home()/".cache"/"opfdata"` → 本机 `C:\Users\sxf51\.cache\opfdata` |
| `model/gridsfm/opfdata_train.py` (`OPFDataAdapterDataset`) | 调用方传入；README 示例用 `~/.cache/opfdata` |
| `model/examples/finetune_opfdata_case6470.ipynb:77` | 硬编码 `/data/OPF/opfdata_pyg_train`（**Linux 路径，Windows/Colab 必须改**） |

下载源在 `OPFDataset.url` 里硬编码：`https://storage.googleapis.com/gridopt-dataset`，
实际拉取的 URL 是 `{url}/{release}/{case_name}_{i}.tar.gz`（`i` ∈ `[0, num_groups)`）。

### 落盘结构（本机实测）

```
<root>/
└── dataset_release_1/                          # 或 dataset_release_1_nminusone（N-1 故障版）
    └── pglib_opf_case14_ieee/
        ├── raw/
        │   ├── pglib_opf_case14_ieee_0.tar.gz      # 下载的分片，别删
        │   └── gridopt-dataset-tmp/
        │       └── dataset_release_1/pglib_opf_case14_ieee/group_0/
        │           └── example_0.json … example_14999.json     # 15,000 个
        └── processed_1/                            # 后缀 = num_groups
            ├── train.pt     35.9 MB    (13,500)
            ├── val.pt        2.0 MB    (   750)
            ├── test.pt       2.0 MB    (   750)
            ├── pre_filter.pt
            └── pre_transform.pt
```

即 `raw_dir = <root>/<release>/<case>/raw`，`processed_dir = <root>/<release>/<case>/processed_<num_groups>`。
**`raw/` 里的 tar.gz 必须保留** —— PyG 靠它判断要不要重新下载；删了就会重下。

---

## 2. 里面都有什么

### 原始 JSON：三个顶层键

`grid`（输入）、`solution`（标签）、`metadata`。以实测的 `example_0.json`（case14）为例：

| 键 | 内容 |
|---|---|
| `grid.context` | 图级特征，`[[[100.0]]]` → baseMVA |
| `grid.nodes.bus` | `14×4` = `[base_kV, type, Vmin, Vmax]` |
| `grid.nodes.generator` | `5×11` = `[mbase, _, Pmin, Pmax, _, Qmin, Qmax, Vg, cp2, cp1, cp0]` |
| `grid.nodes.load` | `11×2` = `[Pd, Qd]` |
| `grid.nodes.shunt` | `1×2` = `[Bs, Gs]` |
| `grid.edges.ac_line` | `senders` / `receivers` / `features` `17×9` |
| `grid.edges.transformer` | `senders` / `receivers` / `features` `3×11` |
| `grid.edges.{generator,load,shunt}_link` | **只有** `senders` / `receivers`，无 features |
| `solution.nodes.bus` | `14×2` = `[theta, V]` ← 电压相角/幅值 |
| `solution.nodes.generator` | `5×2` = `[Pg, Qg]` |
| `solution.edges.{ac_line,transformer}.features` | 线路潮流 |
| `metadata.objective` | 目标函数值，如 `2265.95` |

单文件约 23 KB。每个 `group_*` 目录固定 **15,000** 个 JSON，编号 `0…14999`。

### PyG 加载后变成 `HeteroData`

`OPFDataset.process()`（`torch_geometric/datasets/opf.py:131`）把上面映射成：

- 节点：`bus.x` / `bus.y`，`generator.x` / `generator.y`，`load.x`，`shunt.x`
- 边：`(bus,ac_line,bus)` 带 `edge_attr` + `edge_label`，`(bus,transformer,bus)` 同理，
  以及双向的 `generator_link` / `load_link` / `shunt_link`（无特征）
- 图级：`data.x`（context）、`data.objective`

本机实测 `OPFDataAdapterDataset` 输出的一个图：

```
node types : ['bus', 'generator', 'load', 'shunt']
edge types : 8 个（含双向 link）
bus.x (14, 4)   bus.y (14, 2)
generator.x (5, 11)  generator.y (5, 2)
(bus, ac_line, bus).edge_index (2, 17)   edge_attr (17, 9)
graph-level x : [100.0]
```

### 划分规则

按样本序号切分（`opf.py:207-215`）：`train` 前 90%、`val` 接下来 5%、`test` 最后 5%。
`num_groups=1` → 15000 个 → **13500 / 750 / 750**（本机实测一致）。

### 可用的 case 与分片大小

共 **10 个 case × 20 个分片**，`dataset_release_1`（完整拓扑）和 `dataset_release_1_nminusone`
（N-1 故障拓扑）各一套。**第 0 组**的实测大小：

| case | 单个分片 | `num_groups=20`（PyG 默认）|
|---|---|---|
| pglib_opf_case14_ieee | 26.6 MB | 0.5 GB |
| pglib_opf_case30_ieee | 64.3 MB | 1.3 GB |
| pglib_opf_case57_ieee | 128.4 MB | 2.5 GB |
| pglib_opf_case118_ieee | 303 MB | 5.9 GB |
| **pglib_opf_case500_goc** | **1.15 GB** | 23.0 GB |
| pglib_opf_case2000_goc | 4.48 GB | 89.7 GB |
| pglib_opf_case4661_sdet | 8.90 GB | 178 GB |
| pglib_opf_case6470_rte | 12.9 GB | 258 GB |
| pglib_opf_case10000_goc | 18.5 GB | 369 GB |
| pglib_opf_case13659_pegase | 28.6 GB | 572 GB |

> ⚠️ **只用得到 ≥500 buses 的 case。** `model/README.md` 明确写了 GridSFM-Open 是在 ≥500 母线的电网上训练的，
> case14/30/57/118 属于分布外，结果无意义。所以真正能用的是 case500_goc 及以上。

---

## 3. 如何使用

### 方式 A：GridSFM 自带 CLI（最省事）

```powershell
cd D:\AMyUni\GridSFM\model
python examples\opfdata.py --case pglib_opf_case500_goc --root D:\opfdata --limit 256
# 加了 --gpu 0 就走 GPU；不加就是 CPU
```

`--limit` 限制处理多少个场景，先用小值确认能跑通。完整参数：
`--split {train,val,test}`、`--batch-size`、`--num-groups`、`--ckpt`。

### 方式 B：直接用 PyG（理解数据时）

```python
from torch_geometric.datasets import OPFDataset
ds = OPFDataset(root=r"D:\opfdata", case_name="pglib_opf_case500_goc",
                split="test", num_groups=1)
print(len(ds), ds[0])
```

### 方式 C：微调用的适配器（GridSFM 的 FT 路径）

```python
from gridsfm import OPFDataAdapterDataset
base = OPFDataAdapterDataset(root=r"D:\opfdata", case_name="pglib_opf_case500_goc",
                             variant="fulltop",    # 或 "n1" = N-1 故障
                             split="train", n_graphs=16)
g = base[0]          # HeteroData，直接可送进 SyntheticMixedDataset
```

`variant="n1"` 会自动切到 `dataset_release_1_nminusone`（`OPFDataset(topological_perturbations=True)`）。

---

## 4. 实战注意事项（都是本机踩过的）

1. **下载很慢。** 实测 26.6 MB 花了 **159.5 s ≈ 0.17 MB/s**。按这个速度 case500_goc 的 1.15 GB
   要 **约 2 小时**，case6470_rte 的 12.9 GB 要 **约 21 小时**。务必先只下 `num_groups=1`，
   最好挂机或用 Colab（机房带宽好得多）。
2. **`num_groups` 默认值不一样。** PyG 默认 **20**，GridSFM 的两处封装默认 **1**。
   如果你直接 `OPFDataset(root=...)` 而不传 `num_groups`，会去下 **20 个分片**——
   case500 就是 23 GB。**一定要显式传 `num_groups=1`。**
3. **`n_graphs` 不省下载量。** `opfdata_train.py` 的注释写得很清楚：`n_graphs=16` 也要等完整分片
   下载并解码完，之后才截断。
4. **处理也慢。** 本机 case14（14 母线）处理 15,000 个图花了 **263.5 s**。case500 母线数是它的 ~36 倍，
   处理时间和 `processed_*.pt` 体积都会显著变大（case14 的 train.pt 已经 35.9 MB），
   请预留磁盘（raw + processed 各占一份：本机 case14 的 `raw/` 357 MB + `processed_1/` 40 MB）。
5. **两个路径坑**（`root` 设错会以为"没数据"）：
   - `model/tests/test_ft.py:101,117,150` 和笔记本都硬编码 `/data/OPF/opfdata_pyg_train`，
     在 Windows 会被解析成 `<当前盘>:\data\OPF\...`，导致 6 个 FT 测试**静默 skip**。要让它们真正跑，
     得把 OPFData 放到该路径，或改这几行。
   - 笔记本 `finetune_opfdata_case6470.ipynb` 里要改 `OPFDATA_ROOT`（Colab 见
     [COLAB_GPU_GUIDE.md](COLAB_GPU_GUIDE.md)）。
6. **`release` 目录名会变。** `topological_perturbations=True` 时 release 变成
   `dataset_release_1_nminusone`，而不是同一目录下多出文件——别找错地方。

---

## 5. 最小验证脚本

本机验证这套结构用的脚本（可复用）：

- `D:\AMyUni\gridsfm-winverify\inspect_opfdata.py` — 不用 PyG，直接打印单个 JSON 的全部字段与形状
- `D:\AMyUni\gridsfm-winverify\opfdata_usage_demo.py` — 搭好 `raw/` 布局 → `OPFDataset` → `OPFDataAdapterDataset`，
  并打印三个 split 大小
