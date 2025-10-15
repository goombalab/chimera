# Reassessing the Long-Range Graph Benchmark
Based on the GPS codebase: https://github.com/rampasek/GraphGPS


### Graph MLP Mixer Installation (this is the one that ends up working)
```bash
conda create --name graph_mlpmixer python=3.8
conda activate graph_mlpmixer

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c pyg pytorch-sparse
conda install -c pyg pytorch-scatter
conda install -c pyg pytorch-cluster
conda install -c pyg pyg
pip install ogb
conda install -c conda-forge rdkit
pip install yacs
pip install tensorboard
pip install networkx
pip install einops

# METIS
conda install -c conda-forge metis
pip install metis
pip install torchmetrics
pip install performer-pytorch
pip install wandb

conda clean --all
```

## Command to run Peptides-Func
```
python main.py --cfg configs/GraphSSDGPS/peptides-func-GraphSSDGPS.yaml \
```

## Command to run Peptides-Struct
```
python main.py --cfg configs/GraphSSDGPS/peptides-struct-GraphSSDGPS.yaml
```

## Command to run COCO-SP 
```
python main.py --cfg configs/GraphSSDGPS/coco-GraphSSDGPS.yaml
```

## Command to run VOCS
```
python main.py --cfg configs/GraphSSDGPS/vocsuperpixels-GraphSSDGPS.yaml
```