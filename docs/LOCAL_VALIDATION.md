# 本次开发侧验证（2026-10-21）

本次执行机器是Intel核显、约15.5GiB内存的Windows开发机。用户确认RTX4060Ti16GB/64GB内存训练机是另一台电脑，因此本次准备安装与启动脚本，不在开发机执行CUDA训练。docs/VALIDATION.md保留的是此前另一台训练机的历史记录。

## 已完成

- 自动Python初始化真实执行：固定uv0.13.4的Windows wheel通过SHA256检查，在项目runtime/python内下载CPython3.12.15，创建独立`.venv-tools`。没有修改系统Python，也没有创建或复用GPU训练环境。
- CPU测试环境安装Pillow11.3.0、NumPy1.26.4、toml0.10.2、safetensors0.8.0和Torch2.9.1+cpu，用于数据与坏权重检查；它不能用于桌面CUDA训练。
- 单元测试22项：21项通过，1项跳过。覆盖已有权重导出/NaN拒绝、数据准备、导入前检查、防覆盖、触发词转换、跨风格留出编号一致性和训练锁。
- 跳过项是已安装官方SDXL训练器的dataset schema集成检查；本机没有安装完整GPU训练器和其依赖。
- PowerShell脚本语法检查通过；自动Python初始化实际下载/建环境已验证。
- 六套真实生成数据导入成功，分类各20/15/15/10；保留420张原始PNG的SHA256，生成训练目录副本及当前触发词标注。
- 六种风格分别通过SDXL和Qwen数据校验：12组均为60条有效格式/配对记录。数据格式通过不等于人工质量验收或caption最终核对。
- 六种SDXL数据准备完成，每套53训练+7留出，六套留出原图编号相同。Q版CLI的真实dry-run已执行，生成dataset TOML和命令，没有启动训练器。
- 重复导入验证：0个新增文件，1080个一致文件复用；原图与原始描述不覆盖。

开发侧数据报告在runtime/data-readiness.json，Q版dry-run在runtime/cartoon-dry-run.json；这些本地记录、图片、环境和训练副本不入Git。

## 必须在4060Ti台式机继续验证

尚未执行该台式机上的完整CUDA依赖安装、SDXL底模下载、真实latent/文本缓存、反传、显存峰值或风格质量测试。不能用本次CPU测试替代这些步骤，也没有输出可部署LoRA。

当前按QWEN_DESKTOP.md在台式机执行setup，然后先跑cartoon cache、两步smoke和100步pilot；记录GPU、驱动、Python、Torch、CUDA、缓存与loss，再做正式训练与留出图评估。Jetson候选工作流未被本次迭代激活。

## Qwen优先方案迭代（2026-10-21）

- 默认CLI、安装器和PowerShell入口切到qwen；SDXL保留显式后端选项。
- 最新训练项目29项测试：28通过、1项SDXL专用集成检查跳过；Jetson项目137项测试：126通过、11项平台/真机条件检查跳过。PowerShell语法检查通过。
- 固定Musubi源码已通过本地代理获取并核对revision；未安装GPU训练环境或下载大模型。
- 六套真实配对数据重新准备为schema3编辑指令，每套53训练/7留出，编号一致。420张原始PNG SHA256复核未变。
- 直接执行固定官方源码的ConfigSanitizer类（隔离GPU等重依赖）检查全部六个TOML，通过；三阶段选项名称均与固定源码声明匹配。完整训练器导入/参数解析由GPU主机执行时的check_qwen_backend.py校验，不能当成本机已完整安装Musubi。
- 固定ComfyUI源码的通用Kohya映射分支已对12类Qwen注意力/MLP层名检查；这不是完整模型补丁或推理验收。
- 新增cache独立阶段、100步pilot隔离目录与导出保护、资源采样、Qwen/SDXL架构隔离和层对/rank校验。
- Jetson候选配置激活成功/回执错误拒绝通过临时合成权重测试；生产features.json未改动。
- 本地Jetson公共模型校验如实报三个文件缺失，未声称模型校验通过。没有真实CUDA缓存/反传、显存/时延实测、正式权重或新LoRA真机出图。

报告在runtime/qwen-readiness.json，GPU实际运行会保存backend-check.log、各模式日志/命令、环境快照、TensorBoard与*-resources.json(l)。
