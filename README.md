# SalFoM

This repo is the official PyTorch implementation of ["SalFoM: Dynamic Saliency Prediction with Video Foundation Models"](https://arxiv.org/pdf/2404.03097).
By Morteza Moradi, Mohammad Moradi, Francesco Rundo, Concetto Spampinato, Ali Borji,  and Simone Palazzo.

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
