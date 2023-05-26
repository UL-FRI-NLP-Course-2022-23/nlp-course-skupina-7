from transformers import MT5ForConditionalGeneration, AutoTokenizer, T5ForConditionalGeneration
import torch

model = T5ForConditionalGeneration.from_pretrained("skupina-7/t5-sl-small")
tokenizer = AutoTokenizer.from_pretrained("cjvt/t5-sl-small")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print ("device ",device)
model = model.to(device)

def encode(text):
	context = 'Parafraziraj v slovenščini:'
	encoding = tokenizer.encode_plus(context + text, return_tensors="pt")
	input_ids = encoding["input_ids"].to(device)
	attention_masks = encoding["attention_mask"].to(device)

	return input_ids, attention_masks

def greedy (inp_ids,attn_mask):
	greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=256)
	output = tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
	return output.strip().capitalize()

def topk(inp_ids, attn_mask):
	''' To generate multiple output for same input we are using top-k encoding
	'''
	topkp_output = model.generate(input_ids=inp_ids,
									attention_mask=attn_mask,
									do_sample=True,
									max_length=512,
									top_p=0.84,
									top_k=80,
									num_return_sequences=10,
									min_length=3,
									temperature=0.9,
									repetition_penalty=1.2,
									length_penalty=1.5,
									no_repeat_ngram_size=2,
									)
	outputs = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in topkp_output]
	return [output.strip().capitalize() for output in outputs]

input = 'Glavni slovenski favorit za visoka mesta na domačem evropskem prvenstvu v veslanju na Bledu Rajko Hrvat je v predtekmovanju lahkih skifov zasedel drugo mesto in bo moral v petek ob 10. uri v repasaž.\n'

encoding = encode(input)

print('Original: %s\n\n' % input)
print('Greedy: %s\n\n' % greedy(*encoding))
listk = topk(*encoding)
print('TopK:')
for item in listk:
	print(item)

