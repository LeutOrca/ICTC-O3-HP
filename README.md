# MACE-ICTC-HP

独立的 MACE-ICTC 高性能库，包含笛卡尔 TP、图消息聚合、方向谐波、
SC、通道矩阵及整网训练/求力/AOTI 配置。本仓库仅包含当前源码、API 文档与使用示例，
不包含或依赖 `chorus_o2u1` 包，不分发测试脚本、基准工件或历史源码快照。

**完整 usage：[docs/HP_API.md](docs/HP_API.md)。**
统一导入入口为 `mace_ictc.hp`。已有类与 checkpoint 的 Python 包名
`mace_ictc` 保持兼容，安装包名为 `mace-ictc-hp`。

```python
from mace_ictc.hp import (
    EdgeTensorProduct, GraphMessage, direction_harmonics,
    create_symmetric_contraction, configure_symmetric_contraction, channel_linear, normalized_silu,
    create_model, check_model, configure_model, energy_and_forces,
    prepare_for_export,
)
```

在远端专用环境中安装；本地 Mac 只用于阅读、编辑和 Git：

```bash
python -m pip install -e '.[hp,train]'
python examples/operators.py --lmax 3 --edges 257
python examples/model.py --lmax 3 --layers 3 --layout packed
```

优先使用已验证的 PyTorch 2.7.1/CUDA、Triton、e3nn 0.5.9 环境，CUDA TP JIT
另需兼容的 CUDA toolkit、C++ 编译器和 Ninja。安装到独立环境，避免同名
`mace_ictc` 与公开版/其他 checkout 相互覆盖。A100/集群 CPU 的安装、测试、
示例及编译都要放进 `sbatch`；4090 专机沿用已授权的直接运行例外。

- [算子与整网 API / 支持范围 / 布局 / 训练与部署](docs/HP_API.md)
- [可运行的算子示例](examples/operators.py) · [整网示例](examples/model.py)
- [LAMMPS 接口与正式导出说明](lammps_user_mfftorch/README.md)

默认仍为 qualified `auto` HP + `packed`；`fused_pack` 显式可选。
API 不会自动改精度或冻结已有可训练 TP 系数。公开 MACE-ICTC 仓库保持原样。

独立 TP 可通过构造参数 `execution="legacy"` 使用兼容执行路线，将 `mul_ir`
布局转换放入 TP 的 autograd 边界。它使用本库当前系数和求导实现；默认
`execution="direct"` 保持不变，整网配置不受此独立 TP 选项影响。

```python
op = EdgeTensorProduct.create(
    channels=64, input_lmax=3, edge_lmax=3,
    layout="mul_ir", execution="legacy",
)
```

当前 GPU 路径使用严格 FP32。配置、独立 TP/图消息/通道 GEMM 和 SC 配置
会把 PyTorch 浮点矩阵精度固定为 `highest`，关闭 matmul/cuDNN 的 TF32；
这些进程设置保持关闭，不会自动恢复。张量 dtype、参数和 trainability 保持原样。
高通道（C>=96）且 hidden/edge L<=3 时，默认启用共享元素矩阵 GEMM；
大且较密的图使用按目标原子汇总的 CUDA TP 前向，减少逐边原子加法。
`configure_model(..., wide_channel_policy="off")` 可恢复原 HP 调度。

L4/5 的 eager 整网默认使用 `high_l_policy="auto"`：cor3 采用有界稀疏
SC；cor2 对训练和 L5 非 32 倍数通道推理的非标量输出拆分电路。
高阶 C>=64 的 TP 使用几何计算改写，并按角阶数、cor 和训推阶段选择通道分块。
L5/cor3 推理在 SM 8.9 GPU 保留原分块，训练及 A100 推理使用 32 通道分块。
独立 SC 可用 `configure_symmetric_contraction(sc, training=True)` 选择对应训练调度。
`high_l_policy="off"` 可恢复此前高阶
调度，`policy="reference"` 恢复参考实现。原始学习权重与 checkpoint keys
保持一致；高阶 eager 收益不自动适用于已有 AOTI/LAMMPS 二进制。

单设备 L4/5、cor3 默认在计划验收后让原始稠密 U 常量退出 GPU 常驻。
GPU 分解系数、特征和学习权重保持 GPU FP32；checkpoint keys 和数值保留。
`reference_buffer_policy="device"` 保留常规设备缓冲；分布式/并行封装自动选择
设备缓冲。完整回退、重载和封装用法见 API 文档。

高 L 推理支持有界的 AOTI 路径：L4/5、cor3 的 SC 使用共享稀疏前缀
和解析一阶伴随，TP 按输入阶、路径及通道分块。generated 导出准备默认保留
原始 U 在 CPU，不改变 checkpoint schema；正式 CLI 与 Python API 均提供
`reference_buffer_policy` 的设备驻留对照。只有求力图所需的分解常量和冻结权重
进入新 `.pt2`。旧包需要重新导出并保留对应 `.meta`，才能采用这些算子。

GPU 高性能整网支持用户选择 `channels`、`lmax`、`edge_lmax`、
`num_interactions` 与 `correlation`。通道数为任意正整数，包括非 32 倍数；
阶数没有固定为 L<=3，要求 `0 <= lmax <= edge_lmax`；层数为任意正整数。
本轮正式 HP 覆盖 cor=2、3，兼容 cor=1；cor4/5 保留为独立实验。
不同架构分别生成/缓存内核；结构支持不等于每个高阶组合都经过测试或都更快。

```python
model = create_model(channels=96, lmax=2, num_interactions=4, correlation=2)
configure_model(model, training=True, policy="generated")
```

Benchmark、性能数据、图表、内部测试/验证日志、任务脚本和历史源码
另存于维护者本地档案，不包含在本仓库或安装包中。
本目录的示例仅说明 API 调用，不报告性能曲线。

**许可：新增 HP 实现采用 [PolyForm Noncommercial 1.0.0](LICENSE)，另行提供
[商业授权](COMMERCIAL_LICENSE.md)。** 原已按 MIT 授权的代码及第三方部分保留原权利；
具体范围见 [LICENSE_SCOPE.md](LICENSE_SCOPE.md) 和 [NOTICE](NOTICE)。

原样保留 PolyForm 的全部许可用途：它允许非商业目的，也明确允许教育、公共研究
机构等使用，不因资金来源或资助义务取消该例外。未获该协议许可的商业用途须先
取得权利人的独立书面商业授权；不能把本版说成“任何企业或企业资助项目都必须付费”。
这是 source-available 非商业许可版本，使用条件不能描述为 OSI 开源。

本软件未附训练权重的许可，也不自动授予第三方模型权重、专利或商标使用权。
LAMMPS/GPL 接口和组合二进制的分发须另行核对，见许可范围说明。

当前仅支持原生 ICTC 笛卡尔基路线。
`message_schedule="native"/"roles"` 和设备端 CSR 控制保留。旧 `compact`
调用会明确报错，迁移方式见 API 文档。已有模型结构、权重和 SC 路线不改变。

生成 CG 计划的数值零处理与原库稀疏 TP 的规则对齐，减少构造残差
造成的无效展开。新增 `geometry_schedule="streamed"` 的公开选择，按输入
分量计算原生几何项；整网可显式设置 `message_chunk_edges`。
默认仍保留已有容量/高 L 调度，使用方式与显存边界见 API 文档。

独立低 L TP 默认启用按梯度类型编译的一阶和二阶伴随，并减少图反向
临时缓冲。新增 `PackedGraphMessage` 原生二维图接口和显式整网
按梯度类型选择的 `message_schedule="roles"`（L<=3）方案，完整调用见 API 文档。
对外保持原笛卡尔基、现有模型权重和严格 FP32；实验性调度不自动覆盖整网默认。
