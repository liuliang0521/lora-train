# 数据怎么放

每个风格60张目标风格图：people20、animals15、scenes15、fantasy10；六种风格共360张。只有目标图即可做SDXL风格训练；Qwen Edit路线还要求同名原始照片。

~~~text
datasets/
  cartoon/
    people/001.png       # 目标Q版成品
    people/001.txt       # UTF-8描述，包含iliang_chibi
    animals/cat01.jpg
    animals/cat01.txt
    scenes/forest01.png
    scenes/forest01.txt
    fantasy/robot01.png
    fantasy/robot01.txt
    sources/people/001.jpg  # Qwen编辑训练时必须有真实配对原图
  doubao-3d/...
  miyazaki/...
  jojo/...
  van-gogh/...
  sand-art/...
~~~

init会创建目录，原始图片和caption不被覆盖。每类直接放文件，不再嵌套目录。

## 描述怎么写

描述目标内容，不能只写风格名。保持触发词一致，描述主体、角度、表情、服饰、动作和环境。

- 人像：iliang_chibi, a smiling elderly man with a gray beard, three-quarter view, waist-up, wearing a red jacket, standing on a forest path
- 动物：iliang_chibi, a brown dog looking back, side view, full body, sitting beside a blue cup on a wooden floor
- 静物：iliang_impasto, a white ceramic cup and two books on a wooden table, close-up, warm afternoon light
- 沙画：iliang_sandlight, a rabbit in profile among flowers, sand-yellow highlights and dense sand-yellow shadow grains

caption不能有TODO/占位符；不要把多个风格触发词混在同一份描述。风格图视觉需统一：Q版不要混3D公仔；沙画不要混彩色眼睛或金属雕像。自动描述可辅助，但必须人工核对。

用户计划的年龄、性别、角度、构图与表情都要覆盖。动物避免全是正面猫头像；风景包含远中近景；奇幻不全是同一角色。先保证质量，再凑数量。

## 留出与校验

标准60张中固定53训练、7验证（各类留出2/2/2/1）。按类别、图像内容哈希及种子42确定，验证图不会写入训练TOML。不同checkpoint使用同样验证源图、seed、参数。

工具阻止像素重复，不会自动识别同人连拍、同原图多裁切或视觉质量问题。人工整理近重复，避免同主体近重复跨训练/验证。更改图像、caption、分辨率产生新的数据指纹和缓存目录。

支持PNG/JPEG/WebP单帧，短边至少256、宽高比不超过3。修复损坏文件、缺caption、重复文件名后再训练。prepare保留源数据不动，训练副本转RGB。

## Qwen编辑配对

每张目标图必须有sources/<category>/<同名>.png/jpg：
1. 是风格转换前原图，人物数量、姿态和场景对应目标。
2. 构图需对应；工具拒绝明显宽高比错位，但无法自动判断语义配对质量。
3. caption写目标内容；prepare加入编辑指令。源图产生视觉条件，目标风格图产生损失。
4. 不能拿完全相同图作为输入与目标，这只会教重建，工具会拒绝。
5. 两张不相关图片改成同名也不是合格配对。

当前只有成品图就先走SDXL。不要用调黄滤镜、黑图或目标本身伪造编辑输入。Qwen图像生成LoRA和Edit配对LoRA任务不同，不承诺互换效果。

~~~powershell
.\.venv-sdxl\Scripts\python.exe train.py validate --style cartoon
.\.venv-sdxl\Scripts\python.exe train.py prepare --style cartoon
.\.venv-qwen\Scripts\python.exe train.py validate --backend qwen --style cartoon
~~~
