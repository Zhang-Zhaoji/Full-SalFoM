# SalFoM: Dynamic Saliency Prediction with Video Foundation Models

This repo is forked from the official PyTorch implementation of ["SalFoM: Dynamic Saliency Prediction with Video Foundation Models"](https://arxiv.org/pdf/2404.03097).
By Morteza Moradi, Mohammad Moradi, Francesco Rundo, Concetto Spampinato, Ali Borji,  and Simone Palazzo.

I added some essential script to do inference.

# Abstract

Recent advancements in video saliency prediction (VSP) have
shown promising performance compared to the human visual system,
whose emulation is the primary goal of VSP. However, current state-of-
the-art models employ spatio-temporal transformers trained on limited
amounts of data, hindering generalizability adaptation to downstream
tasks. The benefits of vision foundation models present a potential solu-
tion to improve the VSP process. However, adapting image foundation
models to the video domain presents significant challenges in modeling
scene dynamics and capturing temporal information. To address these
challenges, and as the first initiative to design a VSP model based on
video foundation models, we introduce SalFoM, a novel encoder-decoder
video transformer architecture. Our model employs UnMasked Teacher
(UMT) as feature extractor and presents a heterogeneous decoder which
features a locality-aware spatio-temporal transformer and integrates local
and global spatio-temporal information from various perspectives to pro-
duce the final saliency map. Our qualitative and quantitative experiments
on the challenging VSP benchmark datasets of DHF1K, Hollywood-2
and UCF-Sports demonstrate the superiority of our proposed model in
comparison with the state-of-the-art methods.


## Model weights

The SalFoM uses UMT as its feature encoder, and you can find UMT's pretrained weights on K400 [here](https://huggingface.co/OpenGVLab/UMT/blob/main/single_modality/l16_ptk710_ftk710_ftk400_f16_res224.pth) .

The pretrained weights of SalFoM on DHF1K dataset are available [here](https://studentiunict-my.sharepoint.com/:u:/g/personal/mrdmtz92s11z224o_studium_unict_it/EWurLIAL4aZMsKolSQGgmzQBWk03GIj4u5gAWcVAK4oiWg?e=Nwfcp4).

In order to train and test our model we used the scripts from the [TMFI-Net repository](https://github.com/wusonghe/TMFI-Net).



## Cite
If you find this repository helpful, please cite it using the following BibTeX entry.

```latex
@inproceedings{moradi2025salfom,
  title={Salfom: Dynamic saliency prediction with video foundation models},
  author={Moradi, Morteza and Moradi, Mohammad and Rundo, Francesco and Spampinato, Concetto and Borji, Ali and Palazzo, Simone},
  booktitle={International Conference on Pattern Recognition},
  pages={33--48},
  year={2025},
  organization={Springer}
}
```


