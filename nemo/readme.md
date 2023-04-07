# Setup instructions / cheat sheet for NeMo machine translation models setup

Download the pretrained models from clarin.si:
```bash
curl --remote-name-all https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1736{/slen_GEN_nemo-1.2.6.tar.zst,/ensl_GEN_nemo-1.2.6.tar.zst}
```

Extract them using `tar` and `unzstd`:
```bash
tar --use-compress-program=unzstd -xvf ensl_GEN_nemo-1.2.6.tar.zst
tar --use-compress-program=unzstd -xvf slen_GEN_nemo-1.2.6.tar.zst
```

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

Run translation:
```bash
python translate.py
```