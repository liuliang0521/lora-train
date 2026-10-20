# 本轮验证记录（2026-10-20）

## 本机实际完成

- GPU：NVIDIA GeForce RTX4060Ti，16GB；nvidia-smi驱动610.88。
- 独立环境：.venv-sdxl，Python3.12.14，Torch2.9.1+cu128，CUDA12.8。
- CUDA可用、BF16支持，小矩阵GPU运算通过。
- pip check：No broken requirements found。
- 固定版本SDXL训练入口实际导入及 --help 成功。
- 项目生成的SDXL训练参数通过官方ArgumentParser。
- 实际环境中13项单元测试通过，含官方dataset TOML schema、留出划分、坏图/缺caption/重复、Qwen配对、NaN/架构/LoRA层对及导出保护。
- Windows依赖完整版本记录为config/windows-sdxl.lock.txt；可编辑训练器由config/backends.json固定commit。

测试中的合成纯色图片仅是临时文件，用来检验文件处理和错误分支。它们没有被当成风格数据训练或打包。

## 没有完成、不能替代的检查

用户真实风格图和caption尚未放入datasets，因此未执行真实SDXL缓存/反传、风格训练或生成质量评估。模型下载命令和地址已准备，尚未在本轮下载约7GB SDXL底模及2.5GB ControlNet。

Qwen可选路线已核对官方源码参数，提供独立安装脚本和配对校验；没有安装.venv-qwen，也没有进行16GB显存上的Qwen反传。官方普通Qwen训练显存估算不等同于Edit训练容量保证。

Jetson候选图经过结构/配置测试，未连接Orin进行实际GPU推理。新LoRA不存在，当前业务配置未激活候选profile。

## 用户数据就绪后的首次验收

1. 使用.venv-sdxl执行doctor。
2. 下载/放置同版本SDXL1.0底模。
3. 每风格按20/15/15/10准备图片和描述；validate及prepare。
4. 先跑cartoon两步smoke，记录峰值显存和是否出现非有限loss。
5. 正式训练并比较400/800/1200步等候选。
6. 使用留出源图测试SDXL img2img/LoRA/Canny，不用训练目标图自身充当测试输入。
7. 导出部署包，在Jetson独立工作流验收后再激活。
