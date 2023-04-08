# silence all tqdm progress bars
from platform import platform
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

from typing import Union, List, Dict, Optional, Any
from pydantic import BaseModel, Field
from time import time
from glob import glob
from re import findall
import yaml
import os

import torch
from nemo.core.classes.modelPT import ModelPT
from nemo.utils import logging
import contextlib


if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
    logging.info("AMP enabled!\n")
    autocast = torch.cuda.amp.autocast
else:
    @contextlib.contextmanager
    def autocast():
        yield

from nltk import download, sent_tokenize
download('punkt')

_TEXT_LEN_LIMIT = 5000
_TEXT_SPLIT_THRESHOLD = 1024
_SPLIT_LEN = 512
_use_gpu_if_available = True

class NMTModel(BaseModel):
  class Config:
    arbitrary_types_allowed = True
  tag: str
  nemo: ModelPT
  platform: str
  active: int

models: Dict[str, Dict[str, NMTModel]] = {}

class Model(BaseModel):
  tag: str
  workers: Dict[str,Any]
  features: Optional[Dict[str,Any]]
  info: Optional[Dict[str,Any]]

def translate_text(srclang, tgtlang, text):
  time0 = time()
  if srclang.lower() not in models:
    raise Exception("Invalid source language")
  if tgtlang.lower() not in models[srclang.lower()]:
    raise Exception("Invalid target language")

  text_batch = []
  text_batch_split = []
  if len(text) > _TEXT_SPLIT_THRESHOLD:
    _split_start = len(text_batch)
    _sent = sent_tokenize(text)
    i = 0
    while i < len(_sent):
      j = i+1
      while j < len(_sent) and len(' '.join(_sent[i:j])) < _SPLIT_LEN: j+=1
      if len(' '.join(_sent[i:j])) > _TEXT_SPLIT_THRESHOLD:
        _split=findall(rf'(.{{1,{_SPLIT_LEN}}})(?:\s|$)',' '.join(_sent[i:j]))
        text_batch.extend(_split)
      else:
        text_batch.append(' '.join(_sent[i:j]))
      i = j
    _split_end = len(text_batch)
    text_batch_split.append((_split_start,_split_end))
  else:
    text_batch.append(text)

  logging.debug(f' B: {text_batch}, BS: {text_batch_split}')

  if _use_gpu_if_available and torch.cuda.is_available():
      models[srclang.lower()][tgtlang.lower()].nemo = models[srclang.lower()][tgtlang.lower()].nemo.cuda()

  models[srclang.lower()][tgtlang.lower()].active += 1
  translation_batch = models[srclang.lower()][tgtlang.lower()].nemo.translate(text_batch)
  logging.debug(f' BT: {translation_batch}')
  models[srclang.lower()][tgtlang.lower()].active -= 1

  translation = []
  _start = 0
  for _split_start,_split_end in text_batch_split:
    if _split_start != _start:
      translation.extend(translation_batch[_start:_split_start])
    translation.append(' '.join(translation_batch[_split_start:_split_end]))
    _start = _split_end
  if _start < len(translation_batch):
    translation.extend(translation_batch[_start:])

  result =  ' '.join(translation)

  logging.info(f' R: {result}')
  logging.debug(f'duration: {round(time()-time0,2)}s')

  return result


def initialize():
  time0 = time()
  models: Dict[str, Dict[str, NMTModel]] = {}
  for _model_info_path in glob(f"./models/**/model.info",recursive=True):
    with open(_model_info_path) as f:
      _model_info = yaml.safe_load(f)

    lang_pair = _model_info.get('language_pair', None)
    if lang_pair:
      _model_tag = f"{_model_info['language_pair']}:{_model_info['domain']}:{_model_info['version']}"
      _model_platform = "gpu" if _use_gpu_if_available and torch.cuda.is_available() else "cpu"
      _model_path = f"{os.path.dirname(_model_info_path)}/{_model_info['info']['framework'].partition(':')[-1].replace(':','_')}.{_model_info['info']['framework'].partition(':')[0]}"

      model = ModelPT.restore_from(_model_path,map_location="cuda" if _model_platform == "gpu" else "cpu")
      model.freeze()
      model.eval()

      if lang_pair != f"{model.src_language.lower()}{model.tgt_language.lower()}":
        logging.warning(f"Invalid model.info; language_pair '{lang_pair}', {_model_info['info']['framework'].partition(':')[-1].replace(':','_')}.{_model_info['info']['framework'].partition(':')[0]} '{model.src_language.lower()}{model.tgt_language.lower()}', unloading")
        del model
        continue

      models[model.src_language.lower()] = {}
      models[model.src_language.lower()][model.tgt_language.lower()] = NMTModel(
        tag = _model_tag,
        nemo = model,
        platform = _model_platform,
        active = 0,
      )

  logging.info(f'Loaded models {[ (models[src_lang][tgt_lang].tag,models[src_lang][tgt_lang].platform) for src_lang in models for tgt_lang in models[src_lang] ]}')
  logging.info(f'Initialization finished in {round(time()-time0,2)}s')

  return models

if __name__ == "__main__":
  logging.setLevel(logging.ERROR)
  models = initialize()
  
  en = os.listdir('dataset/en')
  en2sl = os.listdir('dataset/en2sl')
  todo = set(en) - set(en2sl)
  for filename in todo:
    with open('dataset/en/'+filename, 'r') as ifile:
      text = ifile.read()
      translated = translate_text("en", "sl", text)
      with open('dataset/en2sl/'+filename, 'w') as ofile:
        ofile.write(translated)
