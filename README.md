# 六风格 LoRA 训练

独立于 Windows ComfyUI 与 Jetson Python 环境。按用户选择，当前默认采用 **Musubi Tuner / Qwen Image Edit 2511 原图—结果配对 LoRA**，保留 **Kohya sd-scripts / SDXL 1.0** 备选路线。对应 RTX 4060 Ti 16GB、64GB RAM。

先读 [数据说明](docs/DATASET.md)、[环境与训练](docs/TRAINING.md)、[Jetson 技术方案](docs/JETSON_PLAN.md)。

台式机安装与启动按 [Qwen 配对训练指南](docs/QWEN_DESKTOP.md) 执行。[SDXL备选指南](docs/WINDOWS_DESKTOP.md) 保留。当前开发机是Intel核显电脑，不具备CUDA训练能力。历史RTX机器验证见 [验证记录](docs/VALIDATION.md)，本次脚本和数据检查见 [本次开发验证](docs/LOCAL_VALIDATION.md)。

## 快速开始

在已装NVIDIA驱动、Git且有至少约150GB可用SSD空间的4060 Ti台式机执行。先复制原始 `image/`、`train/` 和 `metadata/plan.json`，这些数据不在Git中：

~~~powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Backend qwen -ImportGenerated -DownloadModel
$python = ".\.venv-qwen\Scripts\python.exe"
& $python train.py doctor --backend qwen
~~~

setup通过已校验的uv自动准备项目内Python3.12和独立环境，不修改系统Python或ComfyUI。已有兼容Python可传 `-Python "C:\Python312\python.exe"`。需要代理时显式加 `-Proxy "http://127.0.0.1:7890"`，默认沿用系统网络/现有环境变量。模型、数据、环境和输出均不入Git。

统一启动入口默认只运行两步试训：

~~~powershell
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Style cartoon -Mode smoke
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Style cartoon -Mode pilot
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Style cartoon -Mode train
~~~

默认backend=qwen。`-Mode preview` 只准备数据和命令，`cache`只缓存，`pilot`独立运行100步且不能导出；`-Style all -Mode train` 顺序训练六种风格，中途失败即停。Qwen正式训练默认自动续训：已完成的风格跳过，未完成的风格从当前数据/配置指纹下步数最高的完整state恢复，新风格从零开始。进程锁阻止同时抢占显卡。双击 `scripts/start-training.cmd` 默认只试训Qwen Q版。

中断后重新执行相同命令即可；需要主动停止时按Ctrl+C，等待进程退出再启动：

~~~powershell
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Style all -Mode train
# 明确从头训练六种风格（旧正式结果移到各运行的archive/，可以找回）
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Style all -Mode train -Restart
# 只重训一个风格
powershell -ExecutionPolicy Bypass -File scripts/train-style.ps1 -Style cartoon -Mode train -Restart
~~~

`-Restart` 只归档所选风格当前指纹下的正式 `output/`、`logs/`、正式日志和命令记录；保留数据、模型、缓存、smoke/pilot、其他指纹和导出包。首次重训中断后续跑时不要再加 `-Restart`。默认每100步保存，未保存的进度需重算；100步之前没有状态时从零开始。只有权重而没有完整state时会停止并提示，不静默覆盖。修改数据或profile会产生新指纹，不跨运行自动恢复。`-Resume <同一运行的state目录>` 仍可手动选择Qwen正式断点（必须有配套checkpoint及步数metadata）；SDXL保留原有手动恢复方式。

每种风格60张目标图及同名.txt描述放入 datasets/<style>/ 四类目录，sources/<category>/保存同名真实原图。现有生成数据可用import-generated一次导入：

~~~powershell
& $python train.py validate --style cartoon
& $python train.py prepare --style cartoon
& $python train.py train --style cartoon --dry-run
& $python train.py train --style cartoon --smoke
& $python train.py train --style cartoon
~~~

先对一个风格跑两步smoke检查显存和反传，再正式训练、评估。少量试训可加 --allow-small，但仍需至少4张合格图片，不跳过格式、描述、重复检查。不要同时训练六个风格。

| id | 触发词 | 风格 |
| --- | --- | --- |
| cartoon | iliang_chibi | 二维Q版 |
| doubao-3d | iliang_soft3d | 圆润哑光3D |
| miyazaki | iliang_handpaint | 手绘动画 |
| jojo | iliang_dramaticink | 硬朗线稿、雕塑明暗 |
| van-gogh | iliang_impasto | 厚涂油画 |
| sand-art | iliang_sandlight | 单色透光沙画 |

每种风格单独一个LoRA，配置在config/styles.json；训练参数在config/profiles.json。

## 输出与导出

训练完成后，在训练机双击 `scripts/start-lora-test.cmd`，或执行：

~~~powershell
powershell -ExecutionPolicy Bypass -File scripts/test-loras.ps1
# 比较一个风格的检查点和强度，跑完7张留出图并加入不加载LoRA的对照
powershell -ExecutionPolicy Bypass -File scripts/test-loras.ps1 -Style cartoon -Checkpoints "600,800,1000" -Strengths "0.6,0.85,1.0" -Images 7 -Baseline
# 只检查输入与测试计划，不启动GPU
powershell -ExecutionPolicy Bypass -File scripts/test-loras.ps1 -DryRun
~~~

默认六风格各3张留出原图、最终权重、512尺寸、20步、CFG4、seed42，顺序生成18张。脚本自动连接或启动同级ComfyUI（默认8188），安装本地严格LoRA加载节点，校验权重及实际层匹配，不依赖Jetson节点、不加载Lightning。首次向已经运行的ComfyUI安装节点后需重启ComfyUI，再执行原命令。结果在 `runs/evaluation/<测试指纹>/index.html`，包含原图、目标参考与实际结果；每张图旁JSON记录checkpoint/input哈希、参数、API工作流与执行信息。相同命令跳过已保存结果，超时后重跑会继续等待原任务，不重复提交。ComfyUI进程保留供继续测试；启动日志在 `runtime/lora-test-comfy.log`。

`-Images 7`使用完整留出集；`-Size 1024`测试部署尺寸；`-Url http://127.0.0.1:8189`连接其他本机端口。高级参数 `--comfy-dir`、`--comfy-python`、`--fingerprint` 可通过 `scripts/test_loras.py` 指定。多个数据指纹时必须明确选择。脚本拒绝与训练并行，也拒绝向有活动任务的ComfyUI追加测试。网页用于人工挑选最佳模型，完成出图不自动代表质量通过。

权重位于 runs/qwen/<style>/<数据指纹>/output/*.safetensors。日志、资源采样、训练/验证划分、缓存、完整命令和环境快照在同一指纹目录。smoke/和pilot/均不允许导出上线。

用固定留出输入选择最佳checkpoint后，显式导出（路径用实际值）：

~~~powershell
& $python train.py export --backend qwen --style cartoon --checkpoint "runs\qwen\cartoon\<指纹>\output\<所选模型>.safetensors" --manifest "runs\qwen\cartoon\<指纹>\dataset-manifest.json"
~~~

部署包在 exports/qwen/cartoon/<权重哈希前缀>/，包含权重、SHA256、架构、触发词和数据/环境记录。不会自动把最后的checkpoint当成最好的一版。Jetson先用非Lightning20步候选工作流验收，不承诺可直接用于四步链。

训练器来源固定在config/backends.json；模型来源在config/models.json。trainkit负责数据、命令编排与检查，实际反传由官方训练器执行。

~~~powershell
& $python -m unittest discover -s tests -v
~~~

六套各60张生成数据已接入导入工具，候选文字描述仍需人工核对。本轮未在4060 Ti上执行真实缓存/反传或正式训练，不包含已训练的六种风格权重。
