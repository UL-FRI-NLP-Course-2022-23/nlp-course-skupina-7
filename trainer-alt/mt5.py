import numpy as np
import pandas as pd
import os
import argparse
import logging

import pytorch_lightning as pl

from T5FineTuner import T5FineTuner

logger = logging.getLogger(__name__)
class LoggingCallback(pl.Callback):
	def on_validation_end(self, trainer, pl_module):
		logger.info("***** Validation results *****")
		if pl_module.is_logger():
			metrics = trainer.callback_metrics
			# Log results
			for key in sorted(metrics):
				if key not in ["log", "progress_bar"]:
					logger.info("{} = {}\n".format(key, str(metrics[key])))

	def on_test_end(self, trainer, pl_module):
		logger.info("***** Test results *****")

		if pl_module.is_logger():
			metrics = trainer.callback_metrics

			# Log and save results to file
			output_test_results_file = os.path.join(pl_module._hparams.output_dir, "test_results.txt")
			with open(output_test_results_file, "w") as writer:
				for key in sorted(metrics):
					if key not in ["log", "progress_bar"]:
						logger.info("{} = {}\n".format(key, str(metrics[key])))
						writer.write("{} = {}\n".format(key, str(metrics[key])))

args = dict(
	output_dir="checkpoints", # path to save the checkpoints
	model_name_or_path='skupina-7/t5-sl-small',
	tokenizer_name_or_path='cjvt/t5-sl-small',
	max_seq_length=512,
	learning_rate=3e-4,
	weight_decay=0.0,
	adam_epsilon=1e-8,
	warmup_steps=0,
	train_batch_size=8,
	eval_batch_size=8,
	num_train_epochs=3,
	gradient_accumulation_steps=8,
	n_gpu=1,
	early_stop_callback=False,
	fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
	opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
	max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
	seed=42,
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(   
    save_weights_only=1, every_n_epochs = 1, dirpath = args["output_dir"], filename="checkpoint-{epoch:02d}-L{val_loss:.2f}", monitor="val_loss", mode="min", save_top_k=1
)

train_params = dict(
	accumulate_grad_batches=args["gradient_accumulation_steps"],
	accelerator='gpu',
	devices=args["n_gpu"],
	max_epochs=args["num_train_epochs"],
	precision= 16 if args["fp_16"] else 32,
	gradient_clip_val=args["max_grad_norm"],
	callbacks=[LoggingCallback()],
	enable_checkpointing=False
)

print ("Initialize model")
model = T5FineTuner('dataset.json', args)

trainer = pl.Trainer(**train_params)

print (" Training model")
trainer.fit(model)

print ("Training finished")

print ("Saving model")
model.model.save_pretrained(args["output_dir"])

print ("Saved model")

