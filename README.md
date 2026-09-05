 # ⚡ Faster Projected GAN

This repository contains the PyTorch implementation of Faster Projected GAN in paper *"Faster Projected GAN: Towards Faster
Few-Shot Image Generation"*.

## Installation
Clone repo

```bash
git clone https://github.com/diarimandimby/Faster-Projected-GAN/
cd Faster-Projected-GAN
```

## Using Faster PG in your own project
```
from discriminator import ProjectedGANDiscriminator
D = ProjectedDiscriminator()
```

```
from generator import FasterProjectedGANGenerator
G = FasterProjectedGANGenerator()
```

For a quick start, try the [Colab](https://colab.research.google.com/drive/1szFFNKWGomsLt4-95aFGVxg8suv4J7jK)

## Citation
```bibtex
@misc{liu2021fasterstabilizedgantraining,
      title={Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis}, 
      author={Bingchen Liu and Yizhe Zhu and Kunpeng Song and Ahmed Elgammal},
      year={2021},
      eprint={2101.04775},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2101.04775}, 
}
```
```bibtex
@InProceedings{Sauer2021NEURIPS,
  author         = {Axel Sauer and Kashyap Chitta and Jens M{\"{u}}ller and Andreas Geiger},
  title          = {Projected GANs Converge Faster},
  booktitle      = {Advances in Neural Information Processing Systems (NeurIPS)},
  year           = {2021},
}
```
```bibtex
@misc{wang2024fasterprojectedganfaster,
      title              = {Faster Projected GAN: Towards Faster Few-Shot Image Generation}, 
      author             = {Chuang Wang and Zhengping Li and Yuwen Hao and Lijun Wang and Xiaoxue Li},
      year               = {2024},
      eprint             = {2403.08778},
      archivePrefix      = {arXiv},
      primaryClass       = {cs.CV},
      url                = {https://arxiv.org/abs/2403.08778}, 
}
```
