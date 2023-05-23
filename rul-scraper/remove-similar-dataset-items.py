import json
import nltk
import numpy as np
import matplotlib.pyplot as plt

def bleu_score(candidate, reference):
    candidate = candidate.split()
    reference = reference.split()
    return nltk.translate.bleu_score.sentence_bleu([reference], candidate)

# candidate = "The cat is on the mat"
# reference = "The cat is on the mat"
# score = bleu_score(candidate, reference)
# print(score)


folder_name = "nlp-paraphrases-27k"
new_folder = "new"

num_of_train = 15
num_of_test = 10

all_texts = []

print("Reading train files...")
for i in range(num_of_train):
    with open(f"{folder_name}\\train-{i}.json", "r", encoding="utf-8") as f:
        ff = json.loads(f.read())
        all_texts += ff

prev = len(all_texts)

print("Reading test files...")
for i in range(num_of_test):
    with open(f"{folder_name}\\test-{i}.json", "r", encoding="utf-8") as f:
        ff = json.loads(f.read())
        all_texts += ff
        # print(len(ff))

l_all_t = len(all_texts)
print(l_all_t)
print(f"Train/test split: {prev / l_all_t}")

all_similarities = []

for index, i in enumerate(all_texts):
    print(f"\rGetting similarity score for {index}/{l_all_t}...", end="")
    similarity = bleu_score(candidate=i["en2sl"], reference=i["sl"])
    i["similarity"] = similarity
    all_similarities.append(similarity)

print(" ")
all_similarities = sorted(all_similarities)[::-1]
print(f"{all_similarities[:100]=}")
# all_texts = sorted(all_texts, key=lambda x: x["similarity"])
print(all_similarities[int(len(all_similarities) / 4)])

print(f"Avg similarity: {sum(all_similarities) / len(all_similarities)}")


# num_bins = 40
#
# n, bins, patches = plt.hist(all_similarities, num_bins)
# plt.xlim([0, 1])
# plt.axvline(x=all_similarities[int(len(all_similarities) / 4)], color='r')
#
# plt.show()

final_all_texts = [
    text for text in all_texts if 0 < text["similarity"] < all_similarities[int(len(all_similarities) / 4)]
]
for i in range(len(final_all_texts)):
    assert 0 < final_all_texts[i]["similarity"] < all_similarities[int(len(all_similarities) / 4)]
    del final_all_texts[i]["similarity"]
print(len(final_all_texts))

tt_split = int(3 * len(final_all_texts) / 4)
train_split, test_split = final_all_texts[:tt_split], final_all_texts[tt_split:]
print(len(train_split))
print(len(test_split))

file_size = 690
i = 0
while train_split:
    to_write, train_split = train_split[:file_size], train_split[file_size:]
    full_fn = f"{folder_name}\\{new_folder}\\train-{i}.json"
    with open(full_fn, "w", encoding="utf-8") as f:
        f.write(json.dumps(to_write))
    i += 1

i = 0
while test_split:
    to_write, test_split = test_split[:file_size], test_split[file_size:]
    full_fn = f"{folder_name}\\{new_folder}\\test-{i}.json"
    with open(full_fn, "w", encoding="utf-8") as f:
        f.write(json.dumps(to_write))
    i += 1



