from torch.utils.data import Dataset
import pandas as pd
import torch

class ParaphraseDataset(Dataset):
	def __init__(self, tokenizer, data):
		self.data = data

		self.inputCol = 'en2sl'
		self.targetCol = 'sl'

		self.tokenizer = tokenizer
		self.inputs = []
		self.targets = []

		self._build()

	def __len__(self):
		return len(self.inputs)

	def __getitem__(self, index):
		source_ids = self.inputs[index]["input_ids"].squeeze()
		target_ids = self.targets[index]["input_ids"].squeeze()

		src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
		target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

		return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

	def _build(self):
		for idx in range(len(self.data)):
			input_text,output_text= self.data.loc[idx, self.inputCol],self.data.loc[idx, self.targetCol]

			input_ = "Slovene context: %s" % (input_text)
			target = "%s " %(output_text)

			# tokenize inputs
			tokenized_inputs = self.tokenizer.batch_encode_plus(
				[input_], max_length=256, pad_to_max_length=True, return_tensors="pt"
			)
			# tokenize targets
			tokenized_targets = self.tokenizer.batch_encode_plus(
				[target], max_length=256, pad_to_max_length=True, return_tensors="pt"
			)

			self.inputs.append(tokenized_inputs)
			self.targets.append(tokenized_targets)

def load_dataset(tokenizer, dataset_path):
	dataset = pd.read_json(dataset_path)
	
	train_size = int(0.8 * len(dataset))
	val_size = len(dataset) - train_size

	train, val = torch.utils.data.random_split(dataset, [train_size, val_size])

	train = dataset.loc[train.indices].reset_index(drop=True)
	val = dataset.loc[val.indices].reset_index(drop=True)

	return ParaphraseDataset(tokenizer, train), ParaphraseDataset(tokenizer, val)