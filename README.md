# [CVPR2025] Hyperspectral Pansharpening via Diffusion Models with Iteratively Zero-Shot Guidance
[Jin-Liang Xiao](https://jin-liangxiao.github.io/), Ting-Zhu Huang*, [Liang-Jian Deng](https://liangjiandeng.github.io/), Guang Lin, Zihan Cao, [Chao Li](https://chaoliatriken.github.io/), [Qibin Zhao](https://qibinzhao.github.io/)*

**Paper:** [DM-ZS](https://openaccess.thecvf.com/content/CVPR2025/papers/Xiao_Hyperspectral_Pansharpening_via_Diffusion_Models_with_Iteratively_Zero-Shot_Guidance_CVPR_2025_paper.pdf)

**My Homepage:** https://jin-liangxiao.github.io/

# Main results
We propose a novel guided diffusion scheme with zero-shot guidance and neural spatial-spectral decomposition (NSSD) to iteratively generate an RGB spatial detail image and map the detail image to the target HR-HSI.
- **Framework:**

<div align="center">
<img src=https://github.com/Jin-liangXiao/Jin-liangXiao.github.io/blob/main/assets/img/cvpr_1.png width=60% alt="ipi1"> 
</div>

- **Zero-Shot Guidance:**

<div align="center">
<img src=https://github.com/Jin-liangXiao/Jin-liangXiao.github.io/blob/main/assets/img/cvpr_3.png width=50% alt="ipi2"> 
</div>

- **NSSD:**

<div align="center">
<img src=https://github.com/Jin-liangXiao/Jin-liangXiao.github.io/blob/main/assets/img/cvpr_2.png width=60% alt="ipi2"> 
</div>

# Download the pretrained diffusion model and run
Before running ``Demo_main.py``， please download the pretrained diffusion model [I190000_E97_gen.pth](https://www.dropbox.com/sh/z6k5ixlhkpwgzt5/AAApBOGEUhHa4qZon0MxUfmua?dl=0) provided by [ddpm-cd](https://github.com/wgcban/ddpm-cd) and put the model into *./dm_weight*

**zero-shot guidance**

If you need to calculate other data, please run ``guide_hyperpan.py`` to train the zero-shot network in advance.

This code is based on the [PLRDiff](https://github.com/earth-insights/PLRDiff), [HIR-Diff](https://github.com/LiPang/HIRDiff), and [ZSL](https://github.com/renweidian/ZSL). Thanks for their awesome works!

# Citation
```bibtex
@inproceedings{xiao2025hyperspectral,
  title={Hyperspectral Pansharpening via Diffusion Models with Iteratively Zero-Shot Guidance},
  author={Xiao, Jin-Liang and Huang, Ting-Zhu and Deng, Liang-Jian and Lin, Guang and Cao, Zihan and Li, Chao and Zhao, Qibin},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={12669--12678},
  year={2025}
}
```

# Contact
If you have any questions, please feel free to contact me via ``jinliang_xiao@163.com``
