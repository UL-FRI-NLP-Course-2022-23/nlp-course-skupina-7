from transformers import MT5ForConditionalGeneration, AutoTokenizer
import torch

model = MT5ForConditionalGeneration.from_pretrained("checkpoints")
tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print ("device ",device)
model = model.to(device)

def encode(text):
	context = 'Slovene context:'
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

input = 'Odvzem prostosti je gotovo eden največjih posegov v človekove pravice. Pomislimo samo, kako so nekateri trpeli in že izgubljali potrpljenje, ko smo se bili za kratek čas primorani zadrževati v okvirih lastnih občin. Ali ker smo zaradi zaprtja šol in karanten čas preživljali za toplimi stenami lastnih domov, pa je bilo mnogim tako hudo, da so govorili, da "še malo, pa se jim bo zmešalo". Ob tem si skušajmo predstavljati, kako se počuti človek, ki ga iz svobode nenadoma pripeljejo v nekaj kvadratnih metrov veliko sobo, ki postane njegov nov dom – dom, v katerem se bo lahko zadržal tudi več let ali celo desetletij.'

encoding = encode(input)

print('Greedy:', greedy(*encoding))
print('TopK:', topk(*encoding))

