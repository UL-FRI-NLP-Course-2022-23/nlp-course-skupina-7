from torch.utils.data import Dataset
import pandas as pd
import torch

class ParaphraseDataset(Dataset):
	def __init__(self, tokenizer, data):
		self.data = data

		self.inputCol = 'en2sl'
		self.targetCol = 'sl'

		self.tokenizer = tokenizer

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		input_ = "Parafraziraj v slovenščini:" + self.data[index][self.inputCol]
		target = self.data[index][self.targetCol]

		# tokenize inputs
		tokenized_inputs = self.tokenizer.batch_encode_plus(
			[input_], max_length=256, pad_to_max_length=True, return_tensors="pt"
		)
		# tokenize targets
		tokenized_targets = self.tokenizer.batch_encode_plus(
			[target], max_length=256, pad_to_max_length=True, return_tensors="pt"
		)

		return {"source_ids": tokenized_inputs['input_ids'].squeeze(), "source_mask": tokenized_inputs['attention_mask'].squeeze(),
				"target_ids": tokenized_targets['input_ids'].squeeze(), "target_mask": tokenized_targets['attention_mask'].squeeze()}