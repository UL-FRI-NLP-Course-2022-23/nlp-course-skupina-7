# Setup instructions / cheat sheet for NeMo machine translation models setup

### Extracting models
Create models directory
```bash
mkdir models
cd models
```

Download the pretrained models from clarin.si ([link](https://www.clarin.si/repository/xmlui/handle/11356/1736)):
```bash
curl --remote-name-all https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1736{/slen_GEN_nemo-1.2.6.tar.zst,/ensl_GEN_nemo-1.2.6.tar.zst}
```

Extract them using `tar` and `unzstd`:
```bash
tar --use-compress-program=unzstd -xvf ensl_GEN_nemo-1.2.6.tar.zst
tar --use-compress-program=unzstd -xvf slen_GEN_nemo-1.2.6.tar.zst
```

### Running translation

Create conda enviroment and install dependencies 
```bash
module load Anaconda3

conda create --name nemo python==3.8

conda init bash # optional
conda activate nemo

# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install Cython
pip install nemo_toolkit['all']
```

Run translation with `python translate.py` or run as a slurm job with `sbatch batch.sh`.