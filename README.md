# 视频固定位置水印去除

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Gradio](https://img.shields.io/badge/web-Gradio-orange)](https://www.gradio.app/)

针对"水印固定在画面同一个角落"的短视频，把它修复掉，输出保持原分辨率、H.264 编码、无音轨。

有 Web 界面和命令行两种用法。Web 界面用 Gradio 搭，上传视频后在第一帧上把水印框住，调两个参数就能批量出结果；命令行就是把视频丢进 `input/` 跑批处理。两边共用同一套修复核心，输出效果一样。

> **免责声明**：本工具仅用于处理你自己拥有版权、或已经拿到授权的视频。拿它去别人的作品上抹水印再传播可能侵权，请确认素材来源合法。模型权重版权归各自作者所有，使用前请遵守其许可证。

<!--
## 截图

界面截图和演示动图放在 `docs/` 目录，仓库里暂时是占位的，自己补上就行：

![Web 界面](docs/screenshot.png)

![演示](docs/demo.gif)

（这两张图还没做，怎么生成见 [docs/README.md](docs/README.md)。）
-->

## 功能特性

- **框选即用**：Web 界面上传视频后自动抽第一帧铺到画布上，鼠标拖一个矩形框住水印，实时显示换算到原分辨率的 `x, y, w, h`，带撤销 / 重做 / 清除。
- **实时进度 + 可取消**：处理时状态区实时显示当前是第几个视频、分辨率、帧数、现在走到哪一步（读帧 / 修复 / 编码）和已用时间，配一根渐变进度条；中途点「取消」，当前这一步完成后就停，已跑完的视频保留可下载。结束后列出每个视频的耗时和输出大小。
- **两种修复方式，一行切换**：`lama`（逐帧，快）和 `sttn`（整段视频一起修，背景在动时更自然），在界面上或 `config.yaml` 里换。
- **批量处理**：一次可以传多个 mp4，串行处理，最后打包成一个 zip 下载。
- **只修水印那一小块**：把水印周围的 ROI 裁出来送进模型，再贴回原帧，不整帧进模型。8G 显存的 4060 跑 2K 视频也够。
- **跨分辨率对齐**：框选时记下参考分辨率，`input/` 里视频分辨率不一致时自动等比换算水印区域。
- **消闪后处理**：时间轴中值平滑 + 边缘羽化，压掉修复区域在背景变化时的帧间闪烁。两个都能单独关。
- **LaMa 批量推理**：逐帧调用 LaMa 每次都要等一次 GPU 同步，改成一口气打包 8 张 ROI 送进模型，同一个视频的修复阶段快了一倍多，输出和逐帧逐像素对齐（实测平均差 0.002/255）。批大小按显存自动定，爆显存会自动减半，也能手动填 1 退回逐帧。
- **输出规格固定**：分辨率跟原视频严格一致，`libx264 / crf 18`，无音轨。编码速度档（`slow` / `medium` / `fast`）可选，默认 `medium`。
- **模型权重不进仓库**：用 `download_models.py` 单独下，仓库里只有代码。
- **便携环境**：`setup_portable.bat` 会在项目目录下建一个 `env\` 路径式 conda 环境，整个项目目录（含 env 和 models）拷到另一台 Windows 机器就能跑。

## 环境要求

| 项目 | 版本 / 说明 |
| --- | --- |
| 系统 | Windows |
| 显卡 | NVIDIA（本项目在 RTX 4060 Laptop 8GB、驱动 566.24 上测的） |
| Python | 3.10（conda 虚拟环境） |
| PyTorch | 2.1.2 + CUDA 12.1 |
| 修复后端 | simple-lama-inpainting（LaMa 逐帧）/ STTN（视频修复），另有 `cv2.inpaint` 降级方案 |
| 视频编码 | imageio-ffmpeg 自带的 ffmpeg，**系统不用单独装 FFmpeg** |
| Web 界面 | Gradio 5.x |

没有 N 卡也能跑（`inpaint.device: cpu`），就是慢很多，实际用起来比较难受。

## 安装

### 方式一：双击脚本（推荐）

第一次用：双击 `安装并启动.bat`。它依次做四件事——准备 Python 环境（3.10）、装 PyTorch 2.1.2 + CUDA 12.1 和 `requirements.txt`、下载模型权重、启动 Web 界面。支持断点续跑：环境已存在就跳过创建，PyTorch 已装好就跳过安装，权重已存在就跳过下载，中途失败的话修好网络再双击一次就行。

以后日常用：双击 `启动.bat`，直接拉起 Web 界面，浏览器自动打开 `http://127.0.0.1:7860`。

脚本选环境的顺序是：

1. 项目里的便携环境 `env\.wmr_ready`（`setup_portable.bat` 装好后写的标记）；
2. 找不到就退回 conda 命名环境 `watermark-remover`；
3. 两个都没有，`安装并启动.bat` 会调 `setup_portable.bat` 现建一个便携环境。

都会自己找 conda：先看 PATH，找不到再依次试 `D:\conda`、`C:\ProgramData\Miniconda3` 这些常见位置。每步失败都会停下来给出提示，窗口不会自动关，方便看报错。

不想用命名环境、想把整个项目做成"拷走就能跑"的，直接双击 `setup_portable.bat`，细节见下面的「便携模式」一节。

仓库里还留着一个 `setup.bat`（带逐项环境检查），不想用上面两个的话它也能用。要手动跑命令的话用完整路径：

```
D:\conda\Scripts\conda.exe env create -f environment.yml
```

### 方式二：手动命令

**路线 A：全部走 conda（能顺畅访问 `repo.anaconda.com` 时用）**

```
conda env create -f environment.yml
conda activate watermark-remover
```

**路线 B：conda 只提供 Python，其余走 pip（Anaconda 源连不上时用这条）**

```
D:\conda\Scripts\conda.exe create -n watermark-remover python=3.10 ^
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ --override-channels -y
conda activate watermark-remover
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

这台机器上实测走的是路线 B：`repo.anaconda.com` 时不时连接被重置，而清华的 `pkgs/main` 镜像和 PyTorch 官方的 cu121 wheel 源都很快。`setup.bat` 会先试路线 A，失败自动切到路线 B。

装完验证一下 CUDA 通没通：

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

输出 `True` 就对了。

### 下载模型权重

权重体积大，不随仓库分发，克隆完跑一次这个脚本把它们下到 `models/`：

```
python download_models.py
```

也可以只下其中一个（`--lama` / `--sttn`），或者自己指定 LaMa 的下载地址（`--lama-url <地址>`）。下不动的话看下面「模型权重」一节。

## 便携模式

上面那套 install 走的是 conda 的**命名环境**（`conda env list` 里能看到 `watermark-remover`），它住在 conda 的安装目录下，跟项目是分开的。换台机器就得重装一遍。

便携模式换个思路：把环境建在项目目录里。双击 `setup_portable.bat`，它执行的是

```
conda create -p ./env python=3.10 -y
```

`-p` 是路径式环境，环境本体就是项目下的 `env\` 文件夹。之后不 `conda activate` 也能用，直接：

```
env\python.exe gradio_app.py
```

脚本装完会在 `env\.wmr_ready` 写一个标记文件。`启动.bat` / `安装并启动.bat` / `build_exe.bat` 认的是这个标记，不是 `env\python.exe` —— 因为环境装到一半失败时 `python.exe` 可能已经躺在那儿了，认标记就不会误用半成品。

### 怎么拷到另一台机器

1. 在本机先双击 `setup_portable.bat` 把 `env\` 建好，`download_models.py` 会把权重下到 `models\`；
2. 把整个项目目录（含 `env\`、`models\`）复制到目标机器，路径随便，不用非得一样；
3. 目标机器上装好 conda 或 Miniconda，然后直接双击 `启动.bat`。

**注意**：目标机器**必须先装 conda / Miniconda**。便携环境只是"不占用系统级命名环境"，它本身还是 conda 建出来的，很多包的元数据里记着 conda 的路径，脱离 conda 不一定能跑起来。真要完全脱离 conda，得走 PyInstaller 那条路（见「打包成 exe」），但那条路的坑更多。

如果拷贝后 `启动.bat` 提示找不到可用环境（`.wmr_ready` 没跟着拷过去，或者包路径对不上），在目标机器上重新双击一次 `setup_portable.bat` 就行 —— 它是断点续跑的，`env\` 在就跳过创建，只把缺失的依赖补齐。

### 其他

- `env/` 已经写进 `.gitignore`，不会进仓库。它有几个 GB（torch 一家就 2.5GB 上下），别手滑 commit 进去。
- 便携模式占的磁盘大概是命名环境的两倍多（等于多存一份 torch）。只想在一台机器上用、不打算挪窝的话，用命名环境那条老路就行，没必要建便携环境。
- 建便携环境时要重新下一遍 PyTorch 的 cu121 wheel（约 2.5GB）。如果本机 pip 缓存里已经有了，pip 会直接复用，不会再下。

## 使用方法

### Web 界面

```
conda activate watermark-remover
python gradio_app.py
```

浏览器会自动打开 `http://127.0.0.1:7860`（不想自动开就设一下 `set WMR_NO_BROWSER=1`）。界面按步骤分成四张卡片：

1. **上传视频**：可以多选，只认 `.mp4`。上传后会自动读第一个视频的第一帧，铺到左边的画布上，文件列表直接显示在上传框里。
2. **框选水印**：在画面上**按住鼠标拖一个矩形**把水印框住，下面会实时显示换算到原分辨率的 `x, y, w, h`。框的时候**比水印大一圈**，把外面的白边/描边也一起带上，不然文字去掉了框还在。拖歪了用画布下面的「撤销 / 重做 / 清除」调。
3. **调参数**：参数面板默认收起，展开能改修复方式、掩码膨胀、LaMa 批大小、羽化、时间轴平滑、编码速度这几种，每个参数下面带一行小字说明。默认值直接从 `config.yaml` 读，跟命令行版本一致。图快就把「编码速度」调到 `fast`；`LaMa 批大小`留着「自动」就行。
4. **开始处理**：点「开始处理」后，状态区实时显示当前第几个视频、分辨率、总帧数、走到哪一步（读帧 / 修复 / 贴回编码）、已用时间，配一根渐变进度条。中途不想跑了点「取消」，当前这一步完成后停下，**已经跑完的视频保留**，照样打包下载。跑完出汇总：每个视频的耗时、输出大小，下面能预览第一个结果、单个下载或下整个 zip。

关于这块画布，说明一下为什么不直接用 Gradio 自带的组件：Gradio 的图像编辑器（`gr.ImageEditor`）工具栏里只有**画笔**和**橡皮擦**，**没有矩形工具**；而且 5.50 版本上它前端有个 bug（加载背景图时就抛 `this.history.push is not a function`），undo/redo 也跟着不能用。所以这里的画布是自己用 HTML + JS 写的：一张 `<img>` 铺第一帧，上面盖一个绝对定位的 `div` 当框，鼠标拖动改它的位置和大小，撤销/重做就是自己维护一个栈。框选结果换算回原分辨率后塞进一个隐藏的 Textbox，再当普通输入传给处理函数。

几个实现上的点，用的时候顺便说一下：

- 模型只在第一次点处理时加载，之后一直留在显存里复用，不会每个视频重新加载一遍；改了修复方式或相关参数才会重新加载。
- 处理是 `queue()` 串行的，一次只跑一个任务，避免两个视频抢显存。
- 中间文件写在系统临时目录（`%TEMP%\wmr_web`）里，下一次开始处理或程序退出时会自动清掉，所以结果视频要**在程序关掉之前**下载。
- 区域来源的优先级是：手动坐标 > 画布上的框选 > `config.yaml` 里的 `region`。三个都没有会提示你先框或者填坐标。

### 命令行

命令行保留了原来的流程，适合批量、无人值守地跑。

**第一步：框选水印区域**

把视频放进 `input/`，然后：

```
conda activate watermark-remover
python select_region.py
```

脚本读 `input` 里第一个视频的第一帧，弹出窗口：

1. 按住鼠标左键拖拽，把水印框住；
2. 按回车确认；按 `R` 重新框，按 `ESC` 取消。

视频比屏幕大时窗口会按比例缩小显示，鼠标坐标自动换算回原始分辨率，框选结果不受影响。确认后写进 `config.yaml` 的 `region`，同时在 `output/region_preview.png` 存一张带绿框的预览图，可以打开看看框得准不准。只要水印的相对位置一样，这一个 `config.yaml` 就够所有视频用，不用每个都框。

想指定某个视频来框（比如挑画面最复杂的那个）：

```
python select_region.py input\某个视频.mp4
```

**第二步：批量处理**

```
python main.py
```

把 `input/` 下所有 `.mp4` 挨个处理，结果写到 `output/`，文件名不变。每个视频一条 tqdm 进度条，跑完打印耗时和文件大小。

## 项目结构

```
watermark-remover/
├── gradio_app.py        # Web 界面入口（Gradio）
├── 启动.bat             # 双击启动 Web 界面（优先用 env\ 便携环境）
├── 安装并启动.bat        # 首次使用：建环境 + 装依赖 + 下权重 + 启动
├── setup_portable.bat   # 在项目里建便携环境 env\（可整包拷贝）
├── build_exe.bat        # 用 PyInstaller 打包成 exe（实验性）
├── main.py              # 命令行批量处理入口
├── select_region.py     # 交互式框选水印区域，写入 config.yaml
├── inpaint.py           # 修复调度：裁 ROI / 调后端 / 贴回
├── sttn.py              # STTN 视频修复（网络结构 + 权重加载 + 分段推理）
├── temporal.py          # 修复结果的时间轴平滑，消帧间闪烁
├── utils.py             # 视频读写、区域换算、ffmpeg 编码
├── download_models.py   # 下载 LaMa / STTN 权重到 models/
├── config.yaml          # 配置（水印区域 + 修复参数）
├── environment.yml      # conda 环境依赖
├── requirements.txt     # pip 依赖（备选）
├── setup.bat            # 一键安装脚本
├── docs/                # README 用的截图 / 动图占位
├── env/                 # 便携环境（setup_portable.bat 建，不进 Git）
├── input/               # 待处理视频（不进 Git）
├── output/              # 处理结果（不进 Git）
├── models/              # 模型权重（不进 Git，用脚本下）
├── LICENSE
└── README.md
```

`input` / `output` 不存在的话，`setup.bat` 或 `main.py` 会自动建。

## config.yaml 参数

文件顶部有一份带推荐值的注释，这里再说一遍。

| 参数 | 说明 | 推荐值 |
| --- | --- | --- |
| `region.x/y/w/h` | 水印框位置和大小，由 `select_region.py` 写入 | —— |
| `region.ref_width/ref_height` | 框选时视频的分辨率。水印会随画面缩放，分辨率不一致时靠这两个值等比换算 | 都填上 |
| `inpaint_method` | 修复方式：`lama` / `sttn` / `opencv` | `lama` |
| `inpaint.device` | `auto` / `cuda` / `cpu` | `auto` |
| `inpaint.padding` | ROI 在水印框基础上往外扩的像素。显存紧张就调小 | `20`（紧张时 `8`） |
| `inpaint.mask_dilate` | 掩码膨胀像素，吃掉水印边缘的抗锯齿。外圈剩白边就调大 | `4`（白边残留时 `10~14`） |
| `inpaint.lama_batch_size` | LaMa 一次打包几张 ROI 送进模型。`0` 按显存自动定，`1` 退回逐帧 | `0`（8G 卡自动定成 8） |
| `sttn.segment_size` | STTN 一次同时修多少帧。显存不够会自动对半调小 | `10` |
| `sttn.neighbor_stride` | 相邻窗口步长，越小重叠越多、越稳也越慢 | `5` |
| `sttn.reference_length` | 每个窗口额外参考的帧数 | `10` |
| `sttn.min_segment_size` | 自动下调 `segment_size` 的下限，降到这个值还不够就报错 | `2` |
| `sttn.context_scale` | 送进 STTN 的参考带子比水印框放大多少倍。太小会让掩码占满带子、补出来发糊 | `2`（`2~3`） |
| `sttn.model_path` | 权重路径，留空用 `models/sttn-det.safetensors` | 留空 |
| `output.crf` | H.264 质量，越小画质越好、文件越大 | `18` |
| `output.preset` | 编码速度档位 `slow` / `medium` / `fast` | `medium` |
| `temporal_enable` | 是否开时间轴平滑，用来消帧间闪烁。可单独关 | `true` |
| `temporal_window` | 时间平滑窗口（帧），取奇数，`1` 等于不平滑 | `3` |
| `feather_px` | 修复区贴回时边缘羽化的像素数，`0` 表示不羽化。可单独关 | `8` |

老配置文件里的 `inpaint.backend` 还认：没写 `inpaint_method` 就回落到它。

## LaMa 和 STTN 怎么选

区别在"被水印盖住的那块背景，内容是从哪来的"。

LaMa 是单图修复，每帧单独送进模型，只能靠这一帧里水印周围的花纹去"猜"里面是什么。背景是纯色、渐变或者基本静止的时候猜得挺准；背景一旦在动（镜头平移、树影晃动、车流经过），每帧猜出来的东西不一样，看起来就是水印那一块在糊、在闪。

STTN 是视频修复，把水印附近的一小段帧当成整体，用时空注意力对比前后帧：第 `t` 帧被挡住的位置，会去第 `t-1`、`t+1` 帧里找同一块没被挡住的背景搬过来。所以背景在变的时候，补出来的更接近真实画面。

按素材选：

- **动画 / 插画风、背景变化小** → 用 `lama`。够干净，还快。
- **实拍、纹理丰富、背景在运动** → 用 `sttn`。慢一些，但糊和闪会少。
- 拿不准就先用 `lama` 跑一遍，觉得水印那块发糊再换 `sttn` 对比。

两者还有这些差别：

- **速度**：STTN 要把帧攒成一段一起算（`segment_size` 默认 10 帧一组），窗口之间还有重叠和参考帧。LaMa 改成批量推理之后单帧修复快了一大截，STTN 的差距就更明显了（本机 1080p 240 帧那两个视频：LaMa 热态约 2.3 秒，STTN 20 秒上下）。要速度选 `lama`，要背景运动时的自然度选 `sttn`。
- **显存**：STTN 的注意力是帧数平方级，帧数一多显存涨得快。不够用时程序会自动把 `segment_size` 对半调小重试（10 → 5 → 2），控制台会打印降级提示，不用手动干预。实在还爆就把 `inpaint.padding` 再调小。
- **切换**：改一行配置就行，其它流程不用动。

```yaml
inpaint_method: lama   # 逐帧，快（背景基本静止时够用）
inpaint_method: sttn   # 背景在动，想要更自然（慢一些）
```

有个实测的注意点得说清楚：STTN 的输入尺寸是固定的 432×240，送进去的区域越小、放大得越厉害，补出来的细节就越糊。我们的水印框不大（200 像素上下），所以在本项目那几个测试片（都是扁平配色的动画/插画风）上，STTN 的去干净程度和 LaMa 差不多，边缘还软一点；背景是纯色时两者基本看不出差别。STTN 的优势要等背景本身是实拍、纹理丰富又在运动时才明显。所以别把它当成无脑替换，最好拿自己的一段素材，切一行配置各跑一遍再决定。

时域平滑和羽化跟用哪种修复方式无关，两种模式下都生效，也能单独关：

```yaml
temporal_enable: false   # 不做时间轴平滑
feather_px: 0            # 不羽化
```

## 消闪：时间轴平滑 + 边缘羽化

不管 STTN 还是 LaMa，修复结果在相邻帧之间都可能有一点细微差别。背景一变化——光照在动、纹理在流动——这些差别看起来就是修复区域在闪。STTN 本身用了时间维度的信息，已经比 LaMa 稳不少；还想再压一压就打开时间轴平滑。

`temporal.py` 里的 `temporal_smooth()` 把每帧的 ROI 摞成一个序列，沿时间轴做一次中值滤波（`scipy.ndimage.median_filter`，`size=(window, 1, 1, 1)`），只在时间那一维取窗口，画面的高、宽和颜色通道互不影响。中值的好处是对"某一帧修歪了"这种偶发情况不敏感，不会把误差带到邻居帧上。

`main.py` 的处理顺序是：先逐帧修复、把每帧的 ROI 存进列表，全部处理完再做时间轴平滑，最后把平滑后的 ROI 贴回对应帧。

贴回这一步加了羽化：ROI 四周留一条 `feather_px` 宽的渐变带，从修复结果平滑过渡到原画面，消掉硬贴留下的一圈接缝。水印框外面本来外扩了 `inpaint.padding`（默认 20 像素），所以羽化带离水印还有段距离，不会把水印又露出来。

代价要提前说：因为要先把整个视频的修复结果攒起来再平滑，所有帧都得缓存在内存里。1080p 大约每 100 帧 0.6GB，十秒左右的短片（两三百帧）大概占 1.5~2GB。要是以后处理几万帧的长视频，就得改成按段（chunk）处理，目前没做。

## 模型权重

权重不随仓库分发，`download_models.py` 会帮你下到 `models/`：

```
python download_models.py
```

- **STTN**：`sttn-det.safetensors`，约 66MB，从 hf-mirror 镜像拉（`huggingface.co` 国内基本直连不上，脚本里镜像排在前面）。
- **LaMa**：`big-lama.pt`，约 196MB，从 GitHub Releases 拉。国内经常只有几百 KB/s 甚至被重置，慢的话自己下好放进 `models/`。

装完也可以不跑这个脚本，程序知道去哪找权重：

- `inpaint_method: lama`：**优先用项目里的 `models\big-lama.pt`**。这份在的话，程序启动时会自动把环境变量 `LAMA_MODEL` 指过去（你自己已经设过 `LAMA_MODEL` 的话以你的为准），并打印一行 `[信息] LaMa 权重路径：...` 告诉你实际读的是哪个文件。项目里没有才会退回 torch hub 缓存（`C:\Users\<用户名>\.cache\torch\hub\checkpoints\big-lama.pt`），再没有才联网下。
- `inpaint_method: sttn`：先看环境变量 `STTN_MODEL`，没有就用 `models/sttn-det.safetensors`，再没有自动下。

也就是说，只要 `models\big-lama.pt` 在，运行时就不再依赖 torch hub 缓存那份，也不会重复下 196MB。想换成别处的权重，用环境变量指过去（会盖过项目里那份）：

```
set LAMA_MODEL=D:\models\big-lama.pt
set STTN_MODEL=D:\models\sttn-det.safetensors
```

配合 `config.yaml` 里的 `inpaint_method: opencv` 也能完全绕开权重下载。

## 输出视频规格

- 分辨率严格等于原视频，不缩放、不裁剪；
- 编码 `libx264`，`-crf 18`，像素格式 `yuv420p`；速度档 `-preset` 可配，默认 `medium`；
- 无音轨（`-an`），输出无声视频；
- 加了 `-movflags +faststart`，方便在线播放。

## 打包成 exe

双击 `build_exe.bat` 会用 PyInstaller 把 `gradio_app.py` 打包成 `dist\watermark-remover\` 下的一个文件夹，双击里面的 `watermark-remover.exe` 就能跑。

先说清楚几个取舍，别抱太高期望：

- **用 `--onedir` 不是 `--onefile`**。torch 用 onefile 的话每次启动都要把几个 GB 解压到临时目录，慢得没法用。
- **权重和 ffmpeg 不打进包里**。ffmpeg 走 `imageio-ffmpeg` 包自带的（会被 PyInstaller 收进去），LaMa / STTN 权重靠 `models\` —— `build_exe.bat` 会把现有的 `models\` 拷到 exe 旁边；没拷的话第一次跑会联网下。
- **体积**：torch + gradio 加起来 3GB 上下。
- **兼容性**：打包出来的 exe 带着本机的 CUDA runtime，挪到别的机器上受驱动版本影响，不保证能跑起来。**真的要挪机器，优先用「便携模式」**，那条路稳得多。

exe 启动后一样是浏览器打开 `http://127.0.0.1:7860`，用法和源码跑没区别。`config.yaml` 会被打进包内、同时从 exe 同级目录读，想改配置直接改 exe 旁边那份（`utils.py` 里对打包场景做了路径处理，冻结后 `__file__` 指向临时解包目录，所以项目根改成了按 `sys.executable` 定位）。

如果 PyInstaller 一直报缺模块（gradio 的静态资源和 torch 的 hook 是重灾区），不用死磕，README 里保留这一段就够了，用便携模式。

## 常见问题

**Q：提示 CUDA out of memory，或者显存不够。**

如果用的是 STTN，程序会自动把 `sttn.segment_size` 对半调小重试（10 → 5 → 2），控制台会打印降级提示，一般不用管。要是降到 `sttn.min_segment_size` 还爆，就把 `inpaint.padding` 调小（比如 8）——padding 决定送进模型的带子多大，减半之后显存占用能降不少。

如果用的是 LaMa，先把 `inpaint.padding` 调小（比如 8），还不行就把 `inpaint.device` 改成 `cpu`。LaMa 的自动回退已经做了：某帧爆显存时会把模型挪到 CPU 继续跑完这个视频，只是会慢很多。Web 界面里同一时间只跑一个任务，不会几个视频一起抢显存。

**Q：输出目录、或者 input 里的视频名是中文，会不会出问题？**

不会。读视频时如果 OpenCV 打不开中文路径，会自动把文件复制一份到纯英文的临时目录再读，读完删掉；写视频时如果输出路径含中文，会先编码到英文临时文件再移过去。`utils.py` 里的 `imread_unicode` / `imwrite_unicode` 也是对中文路径安全的读写函数。

**Q：视频分辨率低于 1920×1080。**

会打印一条警告，然后**按原分辨率输出**。这个工具只做水印修复，不做超分辨率放大，没办法把画面"变"成 1080p。等于或高于 1080p 的按原样保持。

**Q：input 里视频分辨率不一样（比如一个 720p、其余 1080p），水印会不会对不准？**

不会。水印跟着画面一起缩放，`select_region.py` 框选时会把当时的分辨率记进 `region.ref_width` / `region.ref_height`，处理每个视频时按 `视频宽度 / ref_width`、`视频高度 / ref_height` 等比换算水印区域。换算结果会打印出来，比如：

```
[信息] 分辨率 1920x1080 与框选时（1280x720）不同，水印区域已换算为 x=1713, y=984, w=136, h=70
```

这一行没出现，多半是老配置里缺 `ref_width` / `ref_height`，重跑一次 `python select_region.py` 重新框选。

Web 界面里框选的视频分辨率就是参考分辨率，会自动用当前上传视频的第一帧尺寸来换算，所以混着不同分辨率的视频一起传也没问题。

**Q：修复出来的区域一闪一闪的，看起来在抖。**

逐帧独立修复（LaMa）有这个问题：每帧单独进模型，结果之间没有约束，背景一动（光照变化、纹理流动）就显得修复区在闪。两个办法：一是把 `inpaint_method` 换成 `sttn`，它本身就参考了前后帧，稳很多；二是保持现状但把 `temporal_enable` 打开（默认就是开的），它会把 ROI 沿时间轴做中值滤波压掉波动，配合 `feather_px: 8` 的边缘羽化，硬边跳变也没了。还觉得闪就把 `temporal_window` 调到 5（窗口越大越稳，但快速运动时可能出现轻微拖影）。

**Q：水印文字没了，但外面还剩一圈白边没去干净。**

说明 `region` 框得偏紧，没把水印自带的那个白色边框一起框进去。水印往往不只是几个字，外面还有圆角矩形描边，描边比文字大一圈。两种改法：一是重新框选，把框拉大一点让描边整个落在框内；二是先把 `inpaint.mask_dilate` 调大（比如 10~14），让修复掩码往外多扩一些，把框外那条描边也盖住。`inpaint.padding` 比 `mask_dilate` 大一截，往外扩一点不会顶到 ROI 边界。

还有个容易忽略的点：`mask_dilate` 和 `padding` 都是按像素算的绝对值，不跟着分辨率缩放。参考分辨率是 720p、实际处理 1440p 的话，这 4 个像素的膨胀相对就"变小"了，白边更容易残留，这时更推荐直接重框。

**Q：`simple-lama-inpainting` 装不上，或者不想下 LaMa 的权重。**

打开 `config.yaml` 把 `inpaint_method` 改成 `opencv`：

```yaml
inpaint_method: opencv
```

这样用 `cv2.inpaint` 的 TELEA 算法来补。纯色或渐变背景还行，背景有纹理的地方会糊一点，属于降级方案。改完直接跑，其它流程不用动。Web 界面的下拉框里也可以选。

**Q：第一次跑卡在 Downloading big-lama.pt 很久，或者直接失败。**

LaMa 权重不在 pip 包里。只要项目 `models\` 下没有 `big-lama.pt`，第一次运行就会去 GitHub Releases 拉一个 196MB 的文件，国内直连经常很慢。

先跑一次 `python download_models.py --lama` 把它下到 `models\big-lama.pt`，之后程序会优先读这份，不用再联网。手动下好放到别处的话，用环境变量指过去也行：

```
set LAMA_MODEL=D:\models\big-lama.pt
python main.py
```

再不行就把 `inpaint_method` 改成 `opencv`，效果差一些但不用下权重。

**Q：`conda env create` 一直卡在 Solving environment，或者报 ReadTimeoutError / SSLEOFError。**

conda 默认读取超时只有 9 秒左右，win-64 的 repodata 体积不小，网速一慢就反复重试、最后失败。`setup.bat` 里已经把超时调高了（120 秒），手动跑可以自己设：

```
set CONDA_REMOTE_READ_TIMEOUT_SECS=120
set CONDA_REMOTE_CONNECT_TIMEOUT_SECS=30
D:\conda\Scripts\conda.exe env create -f environment.yml
```

还不行就换国内镜像：

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
```

注意 TUNA 没有镜像 `nvidia` 频道，`pytorch-cuda=12.1` 还得直连 `conda.anaconda.org`。

**Q：Web 界面上传完视频，画布是空的。**

多半是浏览器没刷新或者上传的文件扩展名不是 `.mp4`（上传框只认 `.mp4`）。换个标准 mp4 编码的文件再试。也可能是视频第一帧读不出来（文件损坏），这时界面上的提示区会写"读不到第一个视频的画面"。

**Q：Web 界面处理完了，但下载按钮点不动 / 文件不见了。**

结果文件放在系统临时目录里，程序退出时会自动清掉。要在关掉程序之前下载。单文件下载和 zip 打包都有。

**Q：需要自己装 FFmpeg 吗？**

不需要。`imageio-ffmpeg` 的 pip 包里自带 ffmpeg 可执行文件，编码直接用它。

**Q：输出视频没有声音？**

对，设计如此。只处理画面，固定加了 `-an`，输出无声视频。

## 性能优化

### 怎么找到瓶颈的

没有直接猜，先临时在 `main.py` 的 `process_one_video()` 里打了几个时间点（读帧裁切 / 模型修复 / 时间轴平滑 / 贴回编码），模型加载也在 `main.py` 和 `gradio_app.py` 里各计了一次时，然后拿 `input/` 里 1080p 240 帧那个视频反复跑，打印出来看。定位完这些打印已经删掉了，只留下 Web 汇总表要用的阶段耗时。

当时（`preset: slow`、LaMa 逐帧）打出来是这样：

```
[耗时] 读帧裁切 1.1s，模型修复 6.2s，时间轴平滑 0.3s，贴回编码 4.2s
```

结论很清楚：**模型修复占了一半以上，编码占三成多**。两个方向都值得动，但修复那块的问题不在算力，在调用方式。于是决定：先把修复的调用方式改掉（收益最大），编码 preset 顺带调默认值。

### 改了什么

**1. LaMa 批量推理（主要改动）**

`inpaint.py` 里的 `repair_patches()` 原来是逐帧调 `SimpleLama`。同一个视频里每帧的 ROI 尺寸完全一样，帧和帧之间又没有依赖（LaMa 是单图模型），所以理论上可以打包成一批一次算完。

打点出来看，243 张 194×122 的 ROI 逐帧要跑 17 秒，绝大部分时间不是花在计算上，而是每调一次模型都要等一次 GPU 同步、再加上 numpy / PIL / tensor 之间的来回搬运。改成一次喂一批：

| 批大小 | 243 帧修复耗时 |
| --- | --- |
| 1（逐帧） | 17.1 秒 |
| 4 | 8.9 秒 |
| 8 | 2.5 秒 |
| 16 | 1.8 秒 |

批大小默认按显存自动定（8G 的 4060 落到 8），`config.yaml` 里 `inpaint.lama_batch_size` 可以写死，写 `1` 就是退回逐帧。显存不够会自动减半重试，减到 1 还不行才报错。

画质这块专门验了：同一段视频分别用逐帧和批量跑一遍，只比 ROI 里的像素：

```
ROI 尺寸 194x122，共 243 帧
平均像素差   0.0023/255  (阈值 < 1/255)
最大像素差   1/255
判定： 通过
```

差异只来自浮点求和的顺序，肉眼不可能看出来。（`simple_lama_inpainting` 自带的 `prepare_img_and_mask` 只支持单张，所以批量这条路是在 `inpaint.py` 里自己拼 tensor 调 `self._lama.model()` 的，掩码补齐方式跟逐帧那条路保持一致。）

**2. 编码 preset 默认从 `slow` 改成 `medium`**

x264 的 `preset` 只影响"为了省码率肯花多少时间算"，CRF 不变的情况下画质基本不变、文件会稍大一点。实测同一个视频：

| preset | 编码耗时 | 输出大小 |
| --- | --- | --- |
| slow | 4.2 秒 | 8.31 MB |
| medium | 3.0 秒 | 8.56 MB |
| fast | 2.6 秒 | 更大一些 |

文件大 3% 换编码快三成，值。Web 界面上加了「编码速度」下拉，命令行改 `config.yaml` 的 `output.preset`。

**3. 前端进度节流**

进度回调原来每帧都 yield 一次，前端拼 HTML、传数据的开销反而成了负担。现在按 0.5 秒节流（`gradio_app.py` 里的 `_UI_REFRESH_SEC`），进度条照样跟手。

**4. 模型加载**

确认了是全局单例：`gradio_app.py` 里按参数算一个 key，参数没变就一直复用同一个 `InpaintEngine`，模型权重留在显存里。验证的时候临时在加载那一行计了时（本机 0.9 秒左右），确认只有第一次处理会走到，之后都命中缓存；这行调试打印已经删了。

**5. 其它**

`gradio_app.py` 里点上传后抽第一帧、算坐标这些活儿都放在后台线程，界面上给了一个"正在读取第一帧"的提示，别让人以为卡死了。这一块没做花哨的优化，解码本来就不慢（见下）。

### 优化前后对比

同一个 1080p / 243 帧视频，四种配置轮转着各跑 3 轮取平均（轮转是为了避开笔记本 GPU 降频带来的抖动）：

| 配置 | 总耗时 | 模型修复 | 贴回编码 | 输出大小 |
| --- | --- | --- | --- | --- |
| 逐帧 + slow（改动前） | 18.5 秒 | 12.0 秒 | 4.5 秒 | 8.31 MB |
| 逐帧 + medium | 13.0 秒 | 7.8 秒 | 3.3 秒 | 8.56 MB |
| 批量 + slow | 8.6 秒 | 2.3 秒 | 4.4 秒 | 8.31 MB |
| **批量 + medium（改动后，Web 默认）** | **7.5 秒** | **2.3 秒** | **3.3 秒** | 8.56 MB |

按最保守的口径（拿"逐帧 + medium"的 12.97 秒当基线）是 42%；按改动前实际在用的配置算 **59%**。CRF 一直是 18，没动。

`input/` 里 4 个视频整体跑 `python main.py`：改动前 58.6 秒，改动后 52.7 秒。单看总时长没上面那个表漂亮，是因为其中两个视频占了冷启动（见下）。

### 试过但不做的

这几个方向都实测过，结论是没收益或者不值得，一并记下来省得以后重复踩：

- **改 `inpaint.padding`**：把 ROI 缩小，修复耗时几乎不动（padding 20 / 16 / 12 / 8 分别是 7.2 / 7.4 / 6.4 / 6.2 秒，padding 4 反而升到 7.5 秒）。说明慢的是每次调用的固定开销，不是 ROI 面积。padding 现在还是保持 20，它管的是"给模型留多少上下文"，不是性能旋钮。
- **`cudnn.benchmark = True`**：开着 5.85 / 5.79 / 5.91 秒，关着 5.74 / 5.85 秒，噪声里看不出差别。不加。
- **时间轴平滑按 chunk 分段**：本来担心内存。实测 ROI 一帧才 69KB，243 帧的修复结果不到 17MB，真正占内存的是留给编码的全帧缓冲（1080p 243 帧约 1.4GB），那是流程结构决定的。分段平滑只会在段边界引入不连续，不值。
- **解码 / CPU 预处理**：整个读帧裁切阶段 1.1 秒（243 帧），已经是顺序读，OpenCV 自己也会开多线程，再多开线程去抢这一秒没意义。不做。

### 一个坑：冷启动

优化完会碰到一个现象：**刚上手第一次处理明显偏慢**。同一个视频连续跑，修复阶段是 `9.0 → 6.9 → 2.4 → 2.0` 秒这样一路掉下来的。原因是笔记本的独显空闲时 SM 频率只有 210MHz，第一次推理才把它拉上来；逐帧路径也有这个现象（冷启动 12.6 秒 vs 热态 6.3 秒）。不是代码问题，第一次跑慢一点属于正常。台式机或者一直有负载的机器上不明显。

（顺带说明：上面所有对比数据都在同一进程里轮转着测的，就是为了让每个配置都处在相近的频率状态，别让冷启动偷偷给某个配置加分。）

### 顺带提一句内存

`main.py` 的流程是"先把整个视频的帧读进内存，最后统一编码"，1080p 每 100 帧大约 0.6GB，十秒左右的短片 1.5~2GB。这是现有架构，没改。要跑几万帧的长视频得改成按段流式处理，目前没做。开了时间轴平滑还要额外攒一份修复结果，但那是 ROI，量很小（上面算过）。

## 免责声明

本项目只用于技术学习和处理自己拥有版权的视频。请勿用于去除他人作品的水印或任何侵权用途。使用者需自行承担因不当使用产生的一切后果。

## License

[MIT](LICENSE)
