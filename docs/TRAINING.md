# 环境、训练和导出

当前优先方案已改为Musubi + Qwen Edit2511配对训练，安装、cache/smoke/pilot/正式训练和内存诊断请按 [QWEN_DESKTOP.md](QWEN_DESKTOP.md)。CLI与PowerShell默认backend=qwen；下面SDXL示例显式指定后端，仅供备选对照。

## 框架

kohya_ss是GUI，SDXL训练核心来自kohya-ss/sd-scripts，本项目直接固定后端commit。Qwen Image Edit采用同为kohya作者维护的Musubi Tuner专用脚本，不把SDXL配置硬套给Qwen。

备选SDXL从768分辨率、batch1、累积4、BF16、梯度检查点、latent和文本缓存、只训练UNet开始。优化器用标准AdamW；bitsandbytes是上游依赖，但SDXL不强制调用其8bit优化器。

首轮六种风格统一rank16/alpha16，后续可将JOJO rank32作为单独对照。学习率8e-5~1e-4；1200~1400 optimizer steps，每200步保存，均为起点。累积4时1200步约4800张次，要比较400/800/1200，防止60张小数据过拟合。

## Windows环境

Python3.10–3.12 x64，建议3.12；Git与NVIDIA驱动。锁定Torch2.9.1/TorchVision0.24.1/cu128，不需要另外安装完整CUDA Toolkit，不使用Jetson ARM包。

~~~powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Backend sdxl -ImportGenerated -DownloadModel
$python = ".\.venv-sdxl\Scripts\python.exe"
& $python train.py doctor --backend sdxl
~~~

setup默认自动准备项目内Python3.12，也可用-Python指定已有3.10–3.12 x64。创建独立.venv-sdxl、安装固定源码和依赖、pip check及CUDA BF16小矩阵。完整freeze保存在runtime/sdxl-freeze.txt，训练另存环境快照。实际GPU训练仍需用户数据的smoke验收。

需要代理时传 -Proxy "http://127.0.0.1:7890"，默认不硬编码代理，沿用系统网络/已有HTTP_PROXY和HTTPS_PROXY。HF_TOKEN只在需要鉴权时设置，不写进仓库。安装和模型缓存默认在runtime目录，不改系统Python路径。

SSD建议SDXL至少50GB空闲、Qwen至少150GB；HF缓存再复制会临时占双份模型空间。64GB RAM并不保证显存足够。先关闭占GPU的ComfyUI/游戏再训练。

## 从数据到训练

~~~powershell
& $python train.py init
& $python train.py download --backend sdxl --model sdxl-base
& $python train.py validate --backend sdxl --style cartoon
& $python train.py train --backend sdxl --style cartoon --dry-run
& $python train.py train --backend sdxl --style cartoon --smoke
& $python train.py train --backend sdxl --style cartoon
~~~

download解析官方模型不可变repo revision，保存大小/SHA回执。已有文件不覆盖，可手工把相同SDXL1.0基座放到config/profiles.json路径。训练不需要ControlNet，结构控制在部署阶段使用。

dry-run会校验数据、准备训练副本、输出真实命令；不启动GPU。smoke执行缓存和两次optimizer step，仅验反传/NaN/OOM，不评价风格质量。数据不齐不会用假数据冒充通过。

缓存开启时不shuffle caption、不用随机裁切/颜色增强；图像、描述或分辨率变化产生新缓存指纹。显存不足先将resolution降640、JOJO rank降16，保留batch1/梯度检查点。基础smoke通过后再试1024并重建缓存。

恢复必须指定同一运行保存的state目录，不是.safetensors：

~~~powershell
& $python train.py train --backend sdxl --style cartoon --resume "runs\sdxl\cartoon\<指纹>\output\<实际state目录>"
~~~

保持原始数据/profile不变。命令在commands.json，日志train.log，smoke日志smoke.log。选择固定验证图上表现更好的checkpoint，不以loss或最后一版自动替代视觉评估。

~~~powershell
& $python train.py export --style cartoon --backend sdxl --checkpoint "runs\sdxl\cartoon\<指纹>\output\<所选模型>.safetensors" --manifest "runs\sdxl\cartoon\<指纹>\dataset-manifest.json"
~~~

exports/<backend>/<style>/<hash>/可复制到Jetson。导出检查有限值、LoRA层对及架构，Jetson加载时再检查实际补丁是否完整匹配。SDXL LoRA不能加载到Qwen。

## 当前优先Qwen Edit 2511

需要真实原图/风格目标配对，且是16GB下待GPU实测的路线。完整当前步骤见QWEN_DESKTOP，以下为底层CLI：

~~~powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Backend qwen -Python "C:\Python312\python.exe"
$python = ".\.venv-qwen\Scripts\python.exe"
& $python train.py download --backend qwen
& $python train.py validate --backend qwen --style cartoon
& $python train.py train --backend qwen --style cartoon --dry-run
& $python train.py train --backend qwen --style cartoon --smoke
& $python train.py train --backend qwen --style cartoon
~~~

先缓存VAE源/目标latent，再缓存视觉文本输出，再训练。512目标/512条件、rank8、batch1、累积4、FP8底模、45层H2D-only CPU交换、ring1、激活CPU卸载、BF16反传；不训练Lightning。新增cache独立阶段、100步pilot与资源采样，试跑权重不允许导出。

官方12GB估算针对普通Qwen图像训练，编辑控制图另占显存，不能当作16GB Edit必定可跑的保证。需实测64GB内存峰值，保留分页空间；512 smoke仍OOM就保持SDXL主线或用更大显存机器，不盲目增加换入层数。

训练输入BF16 DiT、未scaled的qwen_2.5_vl_7b.safetensors。已有ComfyUI的fp8_scaled文本权重不能改名复用。fp8_base表示训练器读取原始权重后量化。现有Qwen20B不会因为LoRA只有几千万参数就变成小底模。

如Windows块交换慢，可在WSL2重新创建Linux训练环境、运行scripts/install_environment.py；不要共享Windows venv、生成的绝对路径TOML或缓存。

## 官方来源与边界

- https://github.com/kohya-ss/sd-scripts
- https://github.com/bmaltais/kohya_ss
- https://github.com/kohya-ss/musubi-tuner/blob/main/docs/qwen_image.md
- https://github.com/kohya-ss/musubi-tuner/blob/main/docs/dataset_config.md
- https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0

至少用留出图、未见人像/动物/复杂背景/白底/透明底，比较3个checkpoint和2个LoRA强度。记录身份保留、主体数量/姿态、背景、风格相似度、耗时、显存和失败案例。

环境检查与脚本测试不等于真实训练完成。已有六套各60对生成数据，候选描述仍待人工复核，本轮尚无已训练完成的六份权重。
