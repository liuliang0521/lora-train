# SDXL备选指南提示

当前优先方案见 [Qwen配对训练](QWEN_DESKTOP.md)。本页保留SDXL备选；由于默认后端已改qwen，所有SDXL启动必须显式加 `-Backend sdxl` 或 `--backend sdxl`。

# 4060 Ti 台式机：安装、数据导入和启动训练

目标机器为Windows x64、RTX4060Ti16GB、64GB内存。当前开发电脑是Intel核显机器；下面的GPU安装和训练命令在另一台台式机执行。不要复制开发机的`.venv-tools`或其他机器的venv。

## 选定方案

| 项目 | 首轮方案 |
| --- | --- |
| 底模 | 官方SDXL1.0，训练和首次部署使用同一底模 |
| 训练器 | config/backends.json固定commit的Kohya sd-scripts |
| 模型组织 | 六个独立LoRA，逐个训练，保留独立触发词 |
| 显存策略 | 768分辨率、batch1、累积4、BF16、梯度检查点、latent/文本磁盘缓存，只训练UNet |
| 参数 | rank16/alpha16；学习率与1200–1400步上限见config/styles.json，每200步保存 |
| 数据 | 每风格60张，53训练+7留出；六风格留出相同原图编号 |
| 后续Jetson | SDXL img2img+对应LoRA，可配SDXL Canny；先隔离出图验收，再激活候选 |

选择SDXL是为了在16GB显存上先建立可检查的训练基线。它和Qwen Edit的身份保持能力不同，不能保证所有输入都更好，也不能将SDXL LoRA挂到Qwen底模。Qwen Edit2511配对训练仍保留为可选实验，首轮不安装、不训练Lightning，也不切换Jetson线上功能。详细部署取舍见JETSON_PLAN.md。

技术依据：[官方SDXL训练说明](https://github.com/kohya-ss/sd-scripts/blob/main/docs/sdxl_train_network.md)、[Qwen编辑训练说明](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/qwen_image.md)、[uv管理Python说明](https://docs.astral.sh/uv/guides/install-python/)。初始参数需要通过实际smoke及留出图验证，不是已实测质量结论。

## 1. 台式机准备

安装NVIDIA显卡驱动和Git for Windows，使用64位PowerShell。单风格建议至少50GB可用SSD；六风格含模型缓存、状态和多个checkpoint建议预留100GB或更多。关闭占GPU的ComfyUI、游戏等程序。训练前可运行`nvidia-smi`确认识别到4060Ti。

从远端更新本仓库后，另行复制数据。Git不会包含图片、模型、环境或训练产物。任选一种数据准备方式：

- 复制生成目录中的`image/`、`train/`、`metadata/plan.json`到台式机仓库相同位置，使用自动导入。
- 已在开发机导入过时，直接复制完整`datasets/`，安装时不传`-ImportGenerated`。
- 数据在单独目录时，安装后用`train.py import-generated --source-root <数据根目录>`导入。

路径可以变化，脚本按自身位置定位仓库。原始标注保留，导入副本会统一训练触发词。先在gallery.html看样本，再核对datasets中的.txt；格式校验不能代替检查身份、背景、肢体和风格一致性。

## 2. 一次安装

在台式机仓库根目录执行：

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -Backend sdxl -ImportGenerated -DownloadModel
~~~

该命令先检查NVIDIA/Git，再执行以下步骤：

1. 按config/bootstrap.json下载固定uv并核对SHA256，由uv在项目内准备Python3.12，创建`.venv-sdxl`；不依赖系统现有Python3.14。
2. 安装固定训练器、CUDA Torch和锁定依赖，执行pip check和GPU BF16小矩阵检查。
3. 初始化目录，导入六套数据及原图配对，下载SDXL底模；不会下载只在部署时使用的Canny模型。
4. 写入runtime/logs/setup-*.log及runtime/sdxl-freeze.txt。

Python由uv使用python-build-standalone提供，具体补丁版本以实际安装记录为准。uv版本、下载地址、校验值固定；训练器和依赖版本由原有配置固定。

需要本机代理时加：

~~~powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -ImportGenerated -DownloadModel -Proxy "http://127.0.0.1:7890"
~~~

已有独立Python3.10–3.12 x64可加`-Python "C:\Python312\python.exe"`。手动放好底模时省略`-DownloadModel`，底模位置见config/profiles.json。由下载工具获取且带有效回执的模型可以验证后复用；已有但无回执的文件不会被覆盖。

已人工改过datasets标注时省略`-ImportGenerated`。相同内容重复导入会跳过；不同内容会拒绝覆盖。网络下载中断可重跑安装；如果Python环境本身未创建完整，先检查并重命名那个不完整环境，再重跑。不要删除原图或人工标注来绕过错误。

## 3. 预览与两步试训

~~~powershell
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Backend sdxl -Style cartoon -Mode validate
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Backend sdxl -Style cartoon -Mode preview
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Backend sdxl -Style cartoon -Mode smoke
~~~

preview只准备训练副本、TOML和真实命令，不运行GPU。smoke会实际缓存、计算梯度并执行两次optimizer step，产物在smoke/，不能导出部署。先检查smoke.log有无OOM、非有限loss或运行失败，再决定正式训练；两步成功不证明风格已经学会。

双击`scripts/start-training.cmd`默认执行Q版smoke并保留终端结果。也可在命令行向这个cmd传入与train-style.ps1相同参数。

## 4. 正式训练与断点恢复

~~~powershell
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Backend sdxl -Style cartoon -Mode train
# 首个风格完成检查后，可逐个执行其他风格，或顺序训练全部：
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Backend sdxl -Style all -Mode train
~~~

风格名为cartoon、doubao-3d、miyazaki、jojo、van-gogh、sand-art。不要在已完成cartoon的相同运行目录里再次从头训练全部；可单独启动尚未完成的风格。脚本遇到已存在checkpoint会停止，避免覆盖。

每个运行在`runs/sdxl/<style>/<fingerprint>/`，包含dataset-manifest.json、commands.json、dataset.toml、environment.json、训练副本和完整日志。正式权重在output/；恢复应指定该目录下真实存在的state目录，而不是safetensors：

~~~powershell
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Backend sdxl -Style cartoon -Mode train -Resume "D:\Project\comfyui\lora_train\runs\sdxl\cartoon\实际指纹\output\实际state目录"
~~~

恢复时数据、标注和profile保持不变。修改后产生新指纹，不能直接接旧state。项目级操作系统锁阻止两个真实训练进程同时运行；进程退出后锁自动释放，无需手动删除锁文件。启动器日志在runtime/logs/launch-*.log，缓存/训练输出另写smoke.log或train.log。

## 5. 看结果、选checkpoint、导出

比较400/800/1200等多个checkpoint，使用固定7张留出源图和额外未见输入，记录身份、背景、风格、耗时、显存、LoRA强度和denoise。训练程序不会自动宣称最后一版权重最好，也不会自动把训练目标图当验证输入。

~~~powershell
$python = ".\.venv-sdxl\Scripts\python.exe"
& $python train.py export --backend sdxl --style cartoon --checkpoint "runs\sdxl\cartoon\实际指纹\output\所选模型.safetensors" --manifest "runs\sdxl\cartoon\实际指纹\dataset-manifest.json"
~~~

包保存在exports/sdxl/cartoon/<权重hash前缀>/，包括权重、SHA256、基座、架构、触发词和数据记录。接入Jetson按JETSON_PLAN.md；六个模型均未完成训练与真机验收前不激活候选功能。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| 安装提示没有NVIDIA | 确认正在4060Ti台式机执行，并先安装NVIDIA驱动；当前核显开发机不能做CUDA训练 |
| 下载GitHub/PyPI/HF失败 | 检查网络，按实际代理端口传-Proxy；不要把密钥写入仓库 |
| venv指向旧目录 | 项目移动后重新创建独立venv；不要复制别台机器的环境 |
| 缺少触发词/分类数量不对 | 使用import-generated接入生成数据，检查datasets结构与对应.txt |
| 导入拒绝覆盖 | 保留人工修订，后续直接validate/prepare；不传-ImportGenerated重装环境 |
| CUDA OOM | 确认batch1，关闭其他GPU程序，先把profiles.json中SDXL分辨率降640，再用新指纹跑smoke |
| 已存在checkpoint | 同一次中断用-Resume；要重新开始则归档旧output/state，再显式重新运行 |
| 两步试训通过但效果不好 | 比较正式checkpoint和留出图，修正训练数据/描述，而不是把smoke权重上线 |

本轮仅验证开发侧代码、数据和启动流程。4060Ti上的驱动、CUDA完整安装、缓存、反传、显存峰值与出图效果，需要在那台台式机实际执行并记录。
