# docs 目录

这里放 README 里引用的截图和演示动图。仓库里目前是空的，需要的话自己补上：

| 文件 | 说明 |
| --- | --- |
| `screenshot.png` | Web 界面截图。建议 1440 宽左右，把上传区、框选区、参数面板都拍进去 |
| `demo.gif` | 演示动图：上传视频 → 框选水印 → 点处理 → 出结果。控制在 5MB 以内，GitHub 才能顺畅预览 |

## 怎么生成

截图直接在浏览器里对着 `http://127.0.0.1:7860` 按 `Win + Shift + S` 截，另存为 `screenshot.png`。

动图可以用 [ScreenToGif](https://www.screentogif.com/)（Windows 上挺好用）或者 ffmpeg 把录屏转成 gif：

```
ffmpeg -i demo.mp4 -vf "fps=12,scale=1000:-1:flags=lanczos" -loop 0 demo.gif
```

放好之后，README 顶部那两行图片引用就能正常显示了。
