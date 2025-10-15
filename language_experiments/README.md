# Chimera

This if the official implementation for Chimera's language experiments.

## About

## Installation
Follow the installation section of [Mamba](https://github.com/state-spaces/mamba); simply,
```bash
pip install mamba-ssm
```

[Option] For training BERT (`./chimera/bert`), install additional required packages via `pip install -r requirements.txt`

## Usage

Our code for training BERT ([./chimera/bert/](./chimera/bert/)) is based on [MosaicBERT](https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert), [M2](https://github.com/HazyResearch/m2), and [Hydra](https://github.com/goombalab/hydra)

Follow the instructions of MosaicBERT ([./chimera/bert/README.md](./chimera/bert/README.md)) for details (*e.g.*, setting up dataset and running code). \
The default configurations are located at:
- Pretrain: [./chimera/bert/yamls/pretrain](./chimera/bert/yamls/pretrain)
- Finetune: [./chimera/bert/yamls/finetune](./chimera/bert/yamls/finetune)

#### Example commands:
Pretrain Chimera on C4 using a single GPU:
```bash
python main.py yamls/pretrain/chimera.yaml
```
Pretrain Chimera on C4 using 8 GPUs:
```bash
composer -n 8 main.py yamls/pretrain/chimera.yaml
```
Finetune Chimera on GLUE:
```bash
python glue.py yamls/finetune/chimera.yaml
```