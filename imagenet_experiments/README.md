# Chimera

This repository provides the official implementations of Chimera which is built on the Structured State Spaces for Sequence Modeling repository.


## Setup

### Requirements
This repository requires Python 3.9+ and Pytorch 1.10+.
It has been tested up to Pytorch 1.13.1.
Other packages are listed in [requirements.txt](./requirements.txt).
Some care may be needed to make some of the library versions compatible, particularly torch/torchvision/torchaudio/torchtext.

Example installation:
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 mamba_ssm -c pytorch -c nvidia
pip install -r requirements.txt
```


## Chimera ViT-B (Table 3)

```
python -m train experiment=s4nd/vit/chimera_b_16_imagenet.yaml
```
