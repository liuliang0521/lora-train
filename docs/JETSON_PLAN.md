# 当前优先：Qwen Edit 2511 配对 LoRA 对接 Jetson

训练见 [QWEN_DESKTOP.md](QWEN_DESKTOP.md)。导出整个 `exports/qwen/<style>/<hash>/` 包到Jetson，包含权重与manifest，不使用smoke/pilot产物。

Jetson项目根目录执行（容器部署则在现有容器内，导出包先放入已挂载目录）：

```bash
python3 scripts/manage.py init
python3 scripts/manage.py download-models --profile qwen
python3 scripts/manage.py verify-models --profile qwen
python3 scripts/manage.py install-lora --package /media/usb/cartoon-export/
# 其余五个风格分别安装
python3 scripts/manage.py lora-profile --backend qwen
```

共用原有BF16 Qwen2511、FP8 scaled视觉文本编码器、Qwen VAE，无需SDXL/ControlNet。安装器使用配置的inference.python，校验完整SHA、触发词、底模、Qwen层名、层对、rank与有限值，不替换NVIDIA Torch。固定安装路径：

```text
data/models/loras/styles/cartoon-qwen.safetensors
data/models/loras/styles/doubao-3d-qwen.safetensors
data/models/loras/styles/miyazaki-qwen.safetensors
data/models/loras/styles/jojo-qwen.safetensors
data/models/loras/styles/van-gogh-qwen.safetensors
data/models/loras/styles/sand-art-qwen.safetensors
```

每份权重旁保留安装manifest。六套独立图在Jetson仓库 `workflows/lora-candidates/qwen/`，功能映射 `config/features.lora-qwen.json`。先在ComfyUI逐个测试，默认20步/CFG4/euler/simple、最长边1024，不加载Lightning或CFGNorm；四步组合需独立测试。主体与已有背景一起风格化，不虚构或替换场景。油画不套名画背景，沙画全图只沙黄深浅。

保持单推理槽，切换与重复出图测共享内存/耗时，不缓存六份完整底模；512训练起点与1024推理质量需对照。新图会改变轮廓，不机械覆盖源alpha。全部验收后，停止新任务、等队列完成并停止服务：

```bash
python3 -m unittest discover -s tests -v
python3 scripts/manage.py lora-profile --backend qwen --activate
python3 scripts/manage.py verify-models
# 按现有Docker/systemd方式启动整个服务，运行doctor并测试业务
```

激活先完整校验，再备份features至runtime/profile-backups并切换；缺失/损坏时不切换。恢复备份并重启可回滚。审核、MySQL、原图12小时、结果主动删除、旧结果和签名密钥保留策略不变。尚未训练或Orin验收，不宣称上线。

## 以下为旧SDXL备选方案记录

## 目标与取舍

六种风格：二维Q版、圆润3D、手绘动画、JOJO、油画、沙画。主线不再依靠20B Qwen加长提示词来承担所有画风变化，而是将风格知识学进六份独立SDXL LoRA。

SDXL UNet约2.6B参数，整套还含CLIP与VAE；明显小于现有Qwen20B，但“较小”不等于所有输入更好。它的身份/语义编辑能力较弱，通过img2img保留颜色和布局，ControlNet约束边缘；Q版大幅改变比例时，保人脸与保风格可能冲突。此处是待验收方案，不承诺在16GB训练机或Orin上已有质量/时延实测结果。

保留Qwen编辑配对LoRA路线，用来对比身份保持和多主体编辑；两种架构的LoRA不互换。

## 主线工作流

~~~text
输入图片
  → 单次按比例缩放，最长边1024，边缘补白至64倍数
  → 可选轻度CPU平滑（降低毛发/噪声碎边）
  ├→ VAE编码，作为img2img初始latent
  └→ Canny边缘（不增加新的分割/深度大模型）
SDXL 1.0 + 所选风格LoRA + 风格触发词
  + SDXL专用Canny ControlNet
  → 一次28步采样
  → VAE解码
  → 沿现有业务链发布结果/缩略图
~~~

轻度平滑是细节降噪，不是语义简化，不能将写实人脸自动变成Q版构造。风格变形来自训练LoRA和适当denoise，而非将原图先糊掉。默认所有原背景参与风格化，不自动抠除或换空白底。

| 风格 | 初始LoRA强度 | denoise | Canny强度/结束比例 | 重点 |
| --- | --- | --- | --- | --- |
| Q版 | 0.85 | 0.62 | 0.25 / 0.50 | 弱结构保布局，允许头身比例变化 |
| 3D | 0.85 | 0.58 | 0.30 / 0.50 | 避免边缘过强变成平面线稿 |
| 手绘 | 0.80 | 0.50 | 0.45 / 0.75 | 保场景、衣着和主体边界 |
| JOJO | 0.80 | 0.55 | 0.40 / 0.75 | 形体、明暗分面，不凭空加肌肉 |
| 油画 | 0.85 | 0.52 | 0.35 / 0.75 | 纹理不是照片加滤镜，禁借名画构图 |
| 沙画 | 0.90 | 0.60 | 0.45 / 0.75 | 用颗粒密度表达层次，保持单色 |

这些是试验起点。需要成对比较denoise ±0.1、LoRA0.6/0.85/1.0及不同checkpoint，不修改一个参数后凭单张图下结论。

SDXL必须配SDXL ControlNet，不能混入SD1.5 ControlNet。训练和推理初期使用同一个SDXL1.0底模；更换DreamShaper/其他融合底模后重新做兼容和质量对照。

## 两阶段仅作为可选对照

如复杂毛发、杂乱背景明显影响画风，可额外试一次Qwen/专门简化模型先简化“纹理”但保留主体身份与场景，然后SDXL LoRA风格化。默认不启用：两轮推理增加时延、共享内存和内容漂移，而且预处理一旦改错身份，后续LoRA无法恢复。

只在固定评估集上证明有收益后再上线双阶段。图片简化不等于先抠背景，不自动删除原有环境。

身份不稳时优先降低denoise/增加真实配对训练，再评估IP-Adapter/参考人脸条件；这些需要新增模型、隐私处理和Orin测试，不在本轮默认依赖中。

## Qwen LoRA候选链

输入 → 原场景编码 → Qwen Edit2511 + 风格LoRA → 20步/CFG4/euler/simple → 解码。

训练底模为非蒸馏Edit2511，因此质量候选链移除Lightning，不直接把普通LoRA叠到4步蒸馏链上。4步Lightning+风格LoRA可做后续独立对照，但要固定checkpoint/强度验证风格和身份，否则不作为默认质量路径。20步是起点，速度预计明显慢于原4步，须在Orin实测，前端2分钟预计不能当SLA。

## 透明图策略

保留源alpha只适合轮廓不变的编辑。Q版/3D可能改变轮廓，直接覆盖原alpha会切掉新发型和头部，因此LoRA候选默认preserve_input_alpha=false。若产品要求“转换后仍透明”，应对已生成结果重新抠图并比较细边质量，而不是机械恢复原mask。当前去背景独立业务不变，API原图审核/队列/名额/结果保留策略均沿用。

## 对接现有Jetson仓库

已增加：
- scripts/generate_lora_profiles.py：生成SDXL/Qwen候选JSON与完整features配置。
- workflows/lora-candidates/sdxl/与qwen/：每架构六套API工作流，SaveImage id=9。
- config/features.lora-sdxl.json及features.lora-qwen.json：备用配置。
- JetsonStylePrepare与JetsonStyleLora节点：尺寸/轻度简化、有限值与完整补丁匹配检查。
- scripts/install_style_lora.py：部署包SHA256、尺寸、架构、触发词检查。

当前生产config/features.json及原六套工作流保持可用。新模型未训练前不自动切换、不以提示词版本偷偷兜底。测试候选请从管理员提交API格式图，手动设置LoadImage文件名与SaveImage前缀，或在隔离测试服务中启用完整候选配置。

## 从训练结果部署

先在训练机执行export。把一个导出目录复制到Jetson，再在Jetson项目中：

~~~bash
python3 scripts/install_style_lora.py --package /media/usb/cartoon-export/
~~~

它将权重安装为data/models/loras/styles/cartoon-sdxl.safetensors并保留manifest。已有同名权重拒绝覆盖，先归档旧版本。六种风格分别安装。

SDXL底模放data/models/checkpoints/sd_xl_base_1.0.safetensors；ControlNet放data/models/controlnet/controlnet-canny-sdxl-1.0-fp16.safetensors。训练仓库download --model sdxl-canny可下载ControlNet，再复制到Jetson。Qwen路线使用现有推理基座/编码器/VAE与新LoRA。

模型文件放齐并完成孤立工作流出图后：

~~~bash
python3 scripts/generate_lora_profiles.py --backend sdxl
# 停止接收新任务，等当前任务完成，再激活
python3 scripts/generate_lora_profiles.py --backend sdxl --activate
# 按本机实际systemd或Docker管理方式重启整个服务
~~~

activate先检查全部模型存在，备份旧features到runtime/profile-backups，再切换。它不代替出图验收、模型哈希与有限值检查。恢复时停止新任务，将备份复制回config/features.json并重启；已入队任务保留原快照，不自动改写。

预热沿现有cartoon功能，激活后会跑新的cartoon LoRA。Orin上逐一测试六种风格切换，关注LoRA补丁/ControlNet缓存累计、峰值共享内存和卸载策略，不并行持有六份模型副本。

## 验收与边界

训练数据默认53训练+7留出，另找至少12张完全未见输入，覆盖人/宠物/风景/静物/奇幻/白底和透明底。记录输入SHA、checkpointSHA、trigger、seed、LoRA强度、denoise、ControlNet参数、耗时、内存峰值。

风格质量、身份保留、场景保留各自打分；重点看Q版是否像二维、3D是否有体积、油画是否有真实笔触、沙画是否保持单色颗粒。颜色提示词不是严格颜色证明，后处理单色映射只能保留亮度纹理，不能替代风格训练；暂不启用强制调色。

缺少用户数据及Jetson远程连接，本轮验收限于环境、脚本参数、数据管线、图编译和坏权重拒绝；不声称六个模型已训练或真机质量已通过。60张是首轮风格启动数据，不能保证所有人像和复杂场景泛化。

本方案不改变现有内容审核与用户数据保留规则。
