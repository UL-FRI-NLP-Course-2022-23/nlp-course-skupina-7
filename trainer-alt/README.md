### Training instructions:
Make sure libraries are installed (pytorch, huggingface transformers, datasets, pytorch lightning)

If on sling:
```
sbatch batch.sh
```

Else:
```
python mt5.py
```

### Inference instructions
You'll also need transformers library and pytorch.

Set the `input` variable to your desired sentence and run
```
python inference.py
```
