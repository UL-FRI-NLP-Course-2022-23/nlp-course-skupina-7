input_file = "mt5-base-rul-pruned_results.json"
grades_file = "grades.json"

from nltk.translate.bleu_score import sentence_bleu

def get_top(top_k, original):
    min_score = 1
    min_text = ""
    for i in top_k:
        score = sentence_bleu([original.split()], i.split())
        if score < min_score:
            min_score = score
            min_text = i
    return min_text


import json

with open(input_file, encoding="utf-8") as f:
    data = json.load(f)

try:
    with open(grades_file, encoding="utf-8") as f:
        grades = json.load(f)
except:
    grades = {}


for i in range(len(data)):
    if str(i) in grades:
        continue

    paraphrases = data[i]

    print(paraphrases["input"], end="")
    print("--------------------------")
    print(paraphrases["greedy"])
    print("")

    vsebina = int(input("Vsebina je enaka: "))
    berljivost = int(input("Berljivost:    "))

    grade = {"greedy": {"vsebina": vsebina, "berljivost": berljivost}}

    top = get_top(paraphrases["topk"], paraphrases["input"])

    print("\n\n")
    print(paraphrases["input"], end="")
    print("--------------------------")
    print(top)
    print("")

    vsebina = int(input("Vsebina je enaka: "))
    berljivost = int(input("Berljivost:    "))

    grade["topk"] = {"vsebina": vsebina, "berljivost": berljivost}

    grades[str(i)] = grade

    quit_ = input("q to quit, enter to continue:")
    if quit_.startswith("q"):
        break

with open(grades_file, "w", encoding="utf-8") as f:
    json.dump(grades, f, indent=4, ensure_ascii=False)

    




    

