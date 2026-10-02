# StreamErase: Ultra-Fast Video Erasing Model

Official PyTorch implementation of the paper:

**StreamErase: Towards Ultra-Fast Streaming Video Erasing**
<br>
[Zhuobai Dong](https://zhuobaidong.github.io/)<sup>1</sup>, 
[Xiaobing Tu](https://scholar.google.com/citations?hl=zh-CN&user=aSqi480AAAAJ)<sup>2</sup>,
[Xiantao Zhang](https://cn.linkedin.com/in/xiantao-zhang-66bb4317/)<sup>2</sup>,
[Jinkui Ren](https://www.linkedin.com/in/jack-ren-2069968/)<sup>2</sup>,
[Yinggui Wang](https://dblp.org/pid/136/1775.html)<sup>2</sup>,
[Yue Ma](https://mayuelala.github.io/)<sup>3</sup>,
[Linfeng Zhang](http://www.zhanglinfeng.tech/)<sup>4</sup>,
<br>
<sup>1</sup>WuHan University, <sup>2</sup>Alibaba Group, <sup>3</sup>The Hong Kong University of Science and Technology, 
<br>
<sup>4</sup>Shanghai Jiao Tong University
<br>
[ArXiv](https://arxiv.org/abs/2512.02899) | [Homepage](https://zhuobaidong.github.io/Glance/) | [Model🤗](https://huggingface.co/CSU-JPG/Glance) | [Demo](https://348d29f48ab953c1e8.gradio.live/)

<img src="assets/teaser.png" alt=""/>

# Quick Start

### 1. Installation
```bash
conda create -n StreamErase python=3.10 -y
conda activate StreamErase
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install flash-attn --no-build-isolation
python setup.py develop
```

### 2. Download Checkpoints
```bash
hf download Wan-AI/Wan2.1-T2V-1.3B  --local-dir wan_models/Wan2.1-T2V-1.3B
hf download zhuobai/StreamErase model.pt --local-dir model_weights
```

### Inference
```bash
python inference.py
```

### Inference Long Video
```bash
python inference_long.py
```

### 测试时间
```bash
python time.py
```

# 制作视频 demo

可参考下面这个链接的 demo.mp4 (demo.py) ，做个类似差不多的，重点要展示我们生成的速度很快，超过实时生成
https://github.com/guandeh17/Self-Forcing

和用户互动的 demo 可参考下面这个链接，目前可以先去除掉 SAM 提取mask的环节，直接让用户选择我们默认提供的 mask
https://huggingface.co/spaces/jixin0101/ObjectClear<br>
https://github.com/sczhou/ProPainter

两个 demo 网页也可以做到一起，一个展示，一个互动

后续制作 mask 可参考下面这个链接
https://github.com/sakshamsingh1/sam3_mask_annotation_tool

