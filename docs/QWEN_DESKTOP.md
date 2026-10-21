# Qwen Edit 2511 配对 LoRA：4060 Ti 台式机操作

当前优先方案是 Musubi Tuner + 非蒸馏 Qwen Image Edit 2511，每种风格一个 LoRA。台式机为 RTX 4060 Ti **16GB**、64GB RAM；Jetson AGX Orin 64GB 负责推理。SDXL 保留为备选，两个架构的 LoRA 不能互换。

## 准备环境与数据

使用 Windows x64，先安装 NVIDIA 驱动和 Git。建议 SSD 至少留 150GB，下载缓存与复制出的模型会临时占双份空间。保留系统分页文件，但频繁换页会拖慢训练，不能替代显存。关闭游戏和其他 GPU 推理进程。

模型和图片不在 Git 中。把已有 `image/`、`train/`、`metadata/plan.json` 一起复制到训练仓库，或复制已整理的 `datasets/`。已有生成数据每风格60对，按样本编号固定53训练、7留出；自动校验不代表人工质量审核完成。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -Backend qwen -ImportGenerated -DownloadModel
```

脚本默认也是 qwen；自动在项目内准备 Python 3.12 与 `.venv-qwen`，固定 Musubi revision、Torch 2.9.1+cu128、TorchVision 0.24.1+cu128；不改系统 Python 或 ComfyUI 环境。需要代理时加 `-Proxy "http://127.0.0.1:7890"`，按实际地址填写。安装日志在 `runtime/logs/`，环境快照在 `runtime/qwen-freeze.txt`。

下载的训练输入为：

```text
models/qwen/qwen_image_edit_2511_bf16.safetensors
models/qwen/qwen_2.5_vl_7b.safetensors
models/qwen/qwen_image_vae.safetensors
```

训练器从 BF16 原始文件加载后做 FP8 量化。不能拿 Jetson 的 `qwen_2.5_vl_7b_fp8_scaled.safetensors` 改名充当训练文本编码器。已有手动模型时省略 `-DownloadModel`，按上表放置同版本文件；下载器不覆盖没有回执的现有模型。

## 分阶段启动

在仓库根目录执行，默认训练 Q版、默认 backend=qwen：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Style cartoon -Mode validate
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Style cartoon -Mode preview
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Style cartoon -Mode cache
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Style cartoon -Mode smoke
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Style cartoon -Mode pilot
```

- validate：检查原图、目标图、caption、重复、配对和数量。
- preview：准备真实训练副本、编辑指令和完整命令，不启动 GPU。
- cache：只运行 VAE 源/目标 latent 缓存和带原图的视觉文本缓存。
- smoke：缓存后做2个 optimizer steps，检查反传与保存。
- pilot：缓存后独立跑100个 optimizer steps，评估持续内存、稳定性和速度。

GPU 执行前会调用固定 Musubi 的真实参数解析器和数据 schema 校验。缓存统一 `--skip_existing`；不同数据、描述、配置、源码 revision 生成不同运行指纹，避免旧缓存混入。缓存阶段用一个读取线程；训练 dataloader 用0个子进程。

启动时自动检查并应用当前固定 Musubi 的小范围兼容补丁：Transformers 5 的 Qwen RMSNorm/解码层 FP8 接口，以及续训步数和进度恢复。补丁逻辑保存在 `trainkit/backend_patches.py`，因此重新克隆项目或安装vendor后仍可恢复；不将整个上游源码入库，不改变固定commit，源码结构不匹配时停止并提示。

smoke、pilot 分别写入 `smoke/`、`pilot/`，禁止导出部署，不与正式 `output/` 混用。双击 `scripts/start-training.cmd` 默认仅执行 Qwen Q版两步 smoke。不要同时训练多个风格。

仅正式训练自动续训；smoke/pilot目录已有权重时，先归档该试跑目录再重新试跑，不覆盖旧检查点。正式训练自动读取配套checkpoint的`ss_steps`，恢复完整权重、优化器、调度器和随机状态；总目标仍是1000步，不会从恢复点额外再跑1000步。恢复时跳过对应epoch中的已训练批次，但不承诺数据shuffle与随机序列逐位复现不中断训练。

## 内存配置与首轮参数

`config/profiles.json` 的 qwen 项：目标分辨率512、control_resolution512、batch1、rank8/alpha8、梯度累积4、AdamW、LR 5e-5、BF16、梯度检查点及激活CPU卸载、FP8底模、45个模块CPU卸载。启用固定版本支持的 H2D-only block swap，ring size=1，减少传输与显存缓冲；这是实验选项，可关闭 `block_swap_h2d_only` 做传统卸载对照。默认不启用 pinned memory，不禁用 mmap，避免额外系统内存压力。

64GB RAM 用来存放卸载权重，不等于80GB显存。官方普通Qwen的12GB估算不含Edit原图条件额外开销；这组参数是待4060 Ti实测的起点。

原图条件分辨率是刻意选择的省显存设置。官方建议 `control_resolution=1024`；512试跑稳定后，质量训练可单独比较目标768、条件1024，修改配置会产生新缓存和运行目录，不能恢复旧 state。不要一次提高全部参数。文本视觉分支由上游按约384面积处理，和用于DiT的原图latent条件不是同一个尺寸参数。

如果H2D-only异常，先关闭该选项重跑；如果显存不足，检查缓存还是反传阶段失败，保留batch1，降低目标/条件尺寸，必要时降rank。不要盲目把卸载层数加到45以上。Windows速度不足可另建WSL2/Linux环境评估，不能复用Windows venv、绝对路径TOML或缓存。

## 日志、持续训练与导出

每个运行位于 `runs/qwen/<style>/<fingerprint>/`。记录 `cache.log`、`smoke.log`、`pilot.log`、`train.log`，各模式有独立 `*-commands.json`。TensorBoard记录在对应 logs 目录；环境和 pip freeze 也会保留。

每3秒采样全系统RAM、可用RAM、swap、GPU已用显存及利用率，写 `*-resources.jsonl` 和汇总 `*-resources.json`，失败也保留摘要。这是全系统采样，可能漏掉瞬时峰值，不能当成精确的单进程 CUDA 峰值。比较速度时使用训练阶段时间，剔除首次下载/缓存；根据实测秒/optimizer step估算总时长，梯度累积4已包含在单步成本中。

通过持续试跑后：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Style cartoon -Mode train
# 单风格效果验收后再顺序训练其他风格
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Style all -Mode train
# 中断后再次执行上一条命令：已完成风格跳过，其余自动续训
# 清空当前正式结果并从头训练（旧结果移至各运行archive/，不删除）
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Style all -Mode train -Restart
# 恢复同一指纹下正式训练保存的state目录，不是safetensors文件
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Style cartoon -Mode train -Resume "runs/qwen/cartoon/<指纹>/output/<实际state目录>"
```

正式训练起点1000 optimizer steps，每100步保存；先比较早、中、后不同检查点，不以最后一版或最低loss自动决定质量。七张留出图不进训练；另加未见图片检查身份、主体数量/姿态、原背景、风格、油画配色和沙黄单色。

按Ctrl+C停止后，等待原进程退出，再执行不带`-Restart`的相同命令。自动恢复会排除缺少文件或被截断的state，回退到较早的完整状态；例如在350步中断，通常从300步恢复，未保存的50步重新计算。首次保存前中断会从零开始。若已有正式结果但找不到可恢复状态，脚本报错，需显式`-Restart`才能重来。

`-Restart`可用于`-Style all`或单风格，仅移动当前数据/配置指纹下的正式`output/`、TensorBoard `logs/`、`train.log`、`train-resources.*`与正式命令记录至同目录`archive/<时间-标识>/`。数据、缓存、模型、smoke/pilot、旧指纹及导出包保留；它不用于释放磁盘空间。该参数不能与`-Resume`或非train模式组合。数据、caption、profile改变时会产生新指纹，不能自动恢复旧运行。已完成的判断以完整最终权重的`ss_steps`达到目标为准，不依赖日志最后一行。

```powershell
.\.venv-qwen\Scripts\python.exe train.py export --backend qwen --style cartoon --checkpoint "runs/qwen/cartoon/<指纹>/output/<所选检查点>.safetensors" --manifest "runs/qwen/cartoon/<指纹>/dataset-manifest.json"
```

导出整个 `exports/qwen/cartoon/<hash>/` 包到 Jetson，含权重、manifest、SHA256、基础模型和训练记录。检查实际权重有限值、层对、rank形状及Qwen架构，不允许smoke/pilot部署。Jetson安装后固定路径为 `data/models/loras/styles/cartoon-qwen.safetensors`，其余风格同理。详见 [Jetson对接](JETSON_PLAN.md)。

先在非Lightning的20步推理链检验LoRA。与现有四步Lightning叠加属于后续对照，不保证兼容、不自动启用。本轮尚未在台式机完成真实GPU缓存/反传，也没有正式训练权重。

依据：[固定版Qwen说明](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/docs/qwen_image.md)、[固定版配对数据配置](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/docs/dataset_config.md)、[模块卸载](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/docs/block_swap.md)。
