# 六风格 LoRA 训练

独立于 Windows ComfyUI 与 Jetson Python 环境。默认采用 **Kohya sd-scripts / SDXL 1.0**，另提供 **Musubi Tuner / Qwen Image Edit 2511** 配对训练路线。对应 RTX 4060 Ti 16GB、64GB RAM。

先读 [数据说明](docs/DATASET.md)、[环境与训练](docs/TRAINING.md)、[Jetson 技术方案](docs/JETSON_PLAN.md)。

本机已安装.venv-sdxl并完成GPU检查，详细边界见 [验证记录](docs/VALIDATION.md)。现在可直接使用该环境放入数据并校验；搬到其他机器时按下文重新安装。

## 快速开始

安装 Python 3.12 x64 后，在 PowerShell 执行（修改实际Python路径）：

~~~powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Backend sdxl -Python "C:\Python312\python.exe"
$python = ".\.venv-sdxl\Scripts\python.exe"
& $python train.py init
& $python train.py download --backend sdxl --model sdxl-base
~~~

setup默认代理为本机127.0.0.1:7890，直连传 -Proxy ""。不要复用现有ComfyUI Python3.13或Anaconda Python3.14。模型、数据、环境和输出均不入Git。

将60张风格成品图及同名.txt描述放入 datasets/<style>/ 四类目录后：

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

权重位于 runs/sdxl/<style>/<数据指纹>/output/*.safetensors。日志、训练/验证划分、缓存、完整命令和环境快照在同一指纹目录。smoke/只用于两步验证，不允许导出上线。

用固定留出输入选择最佳checkpoint后，显式导出（路径用实际值）：

~~~powershell
& $python train.py export --backend sdxl --style cartoon --checkpoint "runs\sdxl\cartoon\<指纹>\output\<所选模型>.safetensors" --manifest "runs\sdxl\cartoon\<指纹>\dataset-manifest.json"
~~~

部署包在 exports/sdxl/cartoon/<权重哈希前缀>/，包含权重、SHA256、架构、触发词和数据/环境记录。不会自动把最后的checkpoint当成最好的一版。

训练器来源固定在config/backends.json；模型来源在config/models.json。trainkit负责数据、命令编排与检查，实际反传由官方训练器执行。

~~~powershell
& $python -m unittest discover -s tests -v
~~~

用户数据仍在准备中，因此本轮不包含已训练的六种风格权重。
