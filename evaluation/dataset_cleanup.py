from similarity import calculate_meteor_score, calculate_bleu_similarity, calculate_cosine_similarity
from datasets import load_dataset

paraphrases = load_dataset("skupina-7/nlp-paraphrases-27k")

treshold = 0.3
count_all = 0
count_remaining = 0
length = len(paraphrases["train"])
paraphrases_filtered = {"train": []}

for i in range(length):

    original = paraphrases["train"][i]["sl"]
    translated = paraphrases["train"][i]["en2sl"]
    similarity = calculate_bleu_similarity(original, translated)

    if similarity < treshold and similarity > 0:
        count_remaining += 1
        paraphrases_filtered["train"].append(paraphrases["train"][i])
    
    count_all += 1
    if count_all % 1000 == 0:
        print(int(count_all/length*100), "%", end="\r")


print("-----------------")
print("Original count: " + str(count_all))
print("Remaining: " + str(count_remaining))
