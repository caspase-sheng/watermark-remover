# 视频固定位置水印去除

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Gradio](https://img.shields.io/badge/web-Gradio-orange)](https://www.gradio.app/)

针对"水印固定在画面同一个角落"的短视频，把它修复掉，输出保持原分辨率、H.264 编码、无音轨。

有 Web 界面和命令行两种用法。Web 界面用 Gradio 搭，上传视频后直接在第一帧上把水印涂掉，调两个参数就能批量出结果；命令行就是把视频丢进 `input/` 跑批处理。两边共用同一套修复核心，输出效果一样。

> **免责声明**：本工具仅用于处理你自己拥有版权、或已经拿到授权的视频。拿它去别人的作品上抹水印再传播可能侵权，请确认素材来源合法。

## 截图

界面截图和演示动图放在 `docs/` 目录，仓库里暂时是占位的，自己补上就行：

![Web 界面](docs/screenshot.png)

![演示](docs/demo.gif)

（这两张图还没做，怎么生成见 [docs/README.md](docs/README.md)。）

## 功能特性

- **框选即用**：Web 界面上传视频后自动抽第一帧，用鼠标把水印涂一下/框一下，实时显示换算到原分辨率的 `x, y, w, h`。
- **两种修复方式，一行切换**：`lama`（逐帧，快）和 `sttn`（整段视频一起修，背景在动时更自然），在界面上或 `config.yaml` 里换。
- **批量处理**：一次可以传多个 mp4，串行处理，最后打包成一个 zip 下载。
- **只修水印那一小块**：把水印周围的 ROI 裁出来送进模型，再贴回原帧，不整帧进模型。8G 显存的 4060 跑 2K 视频也够。
- **跨分辨率对齐**：框选时记下参考分辨率，`input/` 里视频分辨率不一致时自动等比换算水印区域。
- **消闪后处理**：时间轴中值平滑 + 边缘羽化，压掉修复区域在背景变化时的帧间闪烁。两个都能单独关。
- **输出规格固定**：分辨率跟原视频严格一致，`libx264 / crf 18 / preset slow`，无音轨。
- **模型权重不进仓库**：用 `download_models.py` 单独下，仓库里只有代码。

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

### 方式一：一键脚本（推荐）

双击 `setup.bat`。它会创建 conda 环境、装依赖，并逐项检查 PyTorch / CUDA / OpenCV / LaMa。

脚本会自己找 conda：先看 PATH，找不到再依次试 `D:\conda`、`C:\ProgramData\Miniconda3` 这些常见位置。要手动跑命令的话用完整路径：

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

## 使用方法

### Web 界面

```
conda activate watermark-remover
python gradio_app.py
```

浏览器会自动打开 `http://127.0.0.1:7860`（不想自动开就设一下 `set WMR_NO_BROWSER=1`）。界面上从左到右四步：

1. **上传视频**：可以多选，只认 `.mp4`。上传后会自动读第一个视频的第一帧，放到右边的画布上。
2. **框选水印**：在画布上用笔刷涂住水印，或者用矩形工具框住它。涂完下面的框里会显示换算到原分辨率的 `x, y, w, h`。
3. **调参数**：参数面板是折叠的，展开能改修复方式、掩码膨胀、羽化、时间轴平滑这几项。默认值直接从 `config.yaml` 读，跟命令行版本一致。
4. **开始处理**：点按钮，进度条会显示当前第几个视频、总进度。跑完上面出结果预览，右边能单个下载，也能下打包好的 zip。

几个实现上的点，用的时候顺便说一下：

- 模型只在第一次点处理时加载，之后一直留在显存里复用，不会每个视频重新加载一遍；改了修复方式或相关参数才会重新加载。
- 处理是 `queue()` 串行的，一次只跑一个任务，避免两个视频抢显存。
- 中间文件写在系统临时目录（`%TEMP%\wmr_web`）里，下一次开始处理或程序退出时会自动清掉，所以结果视频要**在程序关掉之前**下载。
- 界面上框选的区域优先于 `config.yaml` 里的 `region`。没框选就会用 `config.yaml` 里的，两个都没有会提示先框选。

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
| `sttn.segment_size` | STTN 一次同时修多少帧。显存不够会自动对半调小 | `10` |
| `sttn.neighbor_stride` | 相邻窗口步长，越小重叠越多、越稳也越慢 | `5` |
| `sttn.reference_length` | 每个窗口额外参考的帧数 | `10` |
| `sttn.min_segment_size` | 自动下调 `segment_size` 的下限，降到这个值还不够就报错 | `2` |
| `sttn.context_scale` | 送进 STTN 的参考带子比水印框放大多少倍。太小会让掩码占满带子、补出来发糊 | `2`（`2~3`） |
| `sttn.model_path` | 权重路径，留空用 `models/sttn-det.safetensors` | 留空 |
| `output.crf` | H.264 质量，越小画质越好、文件越大 | `18` |
| `output.preset` | 编码速度档位 | `slow`（想快可改 `medium`） |
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

- **速度**：STTN 要把帧攒成一段一起算（`segment_size` 默认 10 帧一组），窗口之间还有重叠和参考帧，比 LaMa 逐帧慢，具体见下面的性能表。
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

- `inpaint_method: lama`：先看环境变量 `LAMA_MODEL`，没有就去 `模型缓存` 找（`C:\Users\<用户名>\.cache\torch\hub\checkpoints\big-lama.pt`），再没有才会联网下。
- `inpaint_method: sttn`：先看环境变量 `STTN_MODEL`，没有就用 `models/sttn-det.safetensors`，再没有自动下。

用环境变量把路径指过去，跳过自动下载：

```
set LAMA_MODEL=D:\models\big-lama.pt
set STTN_MODEL=D:\models\sttn-det.safetensors
```

配合 `config.yaml` 里的 `inpaint_method: opencv` 也能完全绕开权重下载。

## 输出视频规格

- 分辨率严格等于原视频，不缩放、不裁剪；
- 编码 `libx264`，`-crf 18`，`-preset slow`，像素格式 `yuv420p`；
- 无音轨（`-an`），输出无声视频；
- 加了 `-movflags +faststart`，方便在线播放。

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

LaMa 权重不在 pip 包里，第一次运行会去 GitHub Releases 拉一个 196MB 的文件。国内直连经常很慢。用 `python download_models.py --lama` 自己下一份，或者下好后用环境变量指过去：

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

## 性能参考

本机（RTX 4060 Laptop）跑 `input` 里 4 个真实视频的实测：

| 视频 | 分辨率 | 帧数 | LaMa 用时 | STTN 用时 |
| --- | --- | --- | --- | --- |
| AI-1 | 2560×1440 | 175 | 22.0 秒 | 19.5 秒 |
| AI-2 | 1920×1080 | 243 | 17.8 秒 | 21.9 秒 |
| AI-3 | 1920×1080 | 294 | 12.1 秒 | 23.6 秒 |
| AI-4 | 1920×1080 | 243 | 11.1 秒 | 21.4 秒 |
| 合计 | | | 1 分 3 秒 | 1 分 26 秒 |

两组数字是分开跑的。STTN 每次都攒 `segment_size`（默认 10）帧一起算，帧数越多越慢，整体比 LaMa 慢 40% 左右。两种方式都只处理水印附近的小块，显存占用都不高，真正花时间的大头还是 x264 的 `slow` 档编码。

开了时间轴平滑之后比单纯逐帧多两步：先把所有帧攒在内存里，再多做一次滤波和一次贴回。多出来的耗时是秒级的，可以忽略；主要代价是内存，见上一节。

## 免责声明

本项目只用于技术学习和处理自己拥有版权的视频。请勿用于去除他人作品的水印或任何侵权用途。使用者需自行承担因不当使用产生的一切后果。

## License

[MIT](LICENSE)
