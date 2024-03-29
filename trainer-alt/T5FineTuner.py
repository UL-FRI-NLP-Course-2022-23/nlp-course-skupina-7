import pytorch_lightning as pl
from transformers import (
	AdamW,
	MT5ForConditionalGeneration,
	AutoTokenizer,
	get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
import argparse
import torch

from ParaphraseDataset import ParaphraseDataset
from datasets import load_dataset

class T5FineTuner(pl.LightningModule):
	def __init__(self, dataset_path, hparams):
		super(T5FineTuner, self).__init__()
		self.save_hyperparameters(ignore=["dataset_path"])

		self._hparams = hparams

		self.model = MT5ForConditionalGeneration.from_pretrained(self._hparams["model_name_or_path"])
		self.tokenizer = AutoTokenizer.from_pretrained(self._hparams["tokenizer_name_or_path"])

		dataset = load_dataset("skupina-7/nlp-paraphrases-20k-pruned")
		self.data_train = ParaphraseDataset(self.tokenizer, dataset["train"])
		self.data_val = ParaphraseDataset(self.tokenizer, dataset["test"])

	def is_logger(self):
		return True

	def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
		return self.model(
			input_ids,
			attention_mask=attention_mask,
			decoder_input_ids=decoder_input_ids,
			decoder_attention_mask=decoder_attention_mask,
			labels=labels,
		)

	def _step(self, batch):
		labels = batch["target_ids"]
		labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

		outputs = self(
			input_ids=batch["source_ids"],
			attention_mask=batch["source_mask"],
			labels=labels,
			decoder_attention_mask=batch['target_mask']
		)

		loss = outputs[0]

		return loss

	def training_step(self, batch, batch_idx):
		loss = self._step(batch)

		tensorboard_logs = {"train_loss": loss}
		return {"loss": loss, "log": tensorboard_logs}

	def training_epoch_end(self, outputs):
		avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
		tensorboard_logs = {"avg_train_loss": avg_train_loss}
		self.log("train_loss", avg_train_loss, logger=True, prog_bar=True)
#		return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

	def validation_step(self, batch, batch_idx):
		loss = self._step(batch)
		return {"val_loss": loss}

	def validation_epoch_end(self, outputs):
		avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
		tensorboard_logs = {"val_loss": avg_loss}
		self.log("val_loss", avg_loss, logger=True, prog_bar=True)
#		return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

	def configure_optimizers(self):
		"Prepare optimizer and schedule (linear warmup and decay)"

		model = self.model
		no_decay = ["bias", "LayerNorm.weight"]
		optimizer_grouped_parameters = [
			{
				"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
				"weight_decay": self._hparams["weight_decay"],
			},
			{
				"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
				"weight_decay": 0.0,
			},
		]
		optimizer = AdamW(optimizer_grouped_parameters, lr=self._hparams["learning_rate"], eps=self._hparams["adam_epsilon"])
		self.opt = optimizer
		return [optimizer]

	def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=None, using_lbfgs=None):
		optimizer.step()
		optimizer.zero_grad()
		if optimizer_closure:
			optimizer_closure()
		self.lr_scheduler.step()

	def get_tqdm_dict(self):
		tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

		return tqdm_dict

	def train_dataloader(self):
		dataloader = DataLoader(self.data_train, batch_size=self._hparams["train_batch_size"], drop_last=True, shuffle=True,
								num_workers=4)
		t_total = (
				(len(dataloader.dataset) // (self._hparams["train_batch_size"] * max(1, self._hparams["n_gpu"])))
				// self._hparams["gradient_accumulation_steps"]
				* float(self._hparams["num_train_epochs"])
		)
		scheduler = get_linear_schedule_with_warmup(
			self.opt, num_warmup_steps=self._hparams["warmup_steps"], num_training_steps=t_total
		)
		self.lr_scheduler = scheduler
		return dataloader

	def val_dataloader(self):
		return DataLoader(self.data_val, batch_size=self._hparams["eval_batch_size"], num_workers=4)
