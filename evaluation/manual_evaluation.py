input_file = "mt5-base-rul-pruned_results.json"
grades_file = "grades.json"

# All grades should be between 1 and 5
# ustreznost: 1-5 (kako podoben je pomen parafraze)
# berljivost: 1-5 (kako berljiva je parafraza)
# enakost: 1-5 (kako enaka sta teksta)


from nltk.translate.bleu_score import sentence_bleu


def get_top(top_k, original):
    min_score = 100000
    min_text = ""
    scores = []

    for i in top_k:
        bleu_score = sentence_bleu([original.split()], i.split())

        scores.append(bleu_score)

    sorted_scores = sorted(scores, reverse=True)
    middle = sorted_scores[len(sorted_scores) // 2]

    for i in scores:
        if middle == i:
            return top_k[scores.index(i)]

    



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

    ustreznost = int(input("Ustreznost: "))
    berljivost = int(input("Berljivost: "))
    enakost = int(input("Enakost: "))

    grade = {"greedy": {"ustreznost": ustreznost, "berljivost": berljivost, "enakost": enakost}}

    top = get_top(paraphrases["topk"], paraphrases["input"])

    print("\n\n")
    print(paraphrases["input"], end="")
    print("--------------------------")
    print(top)
    print("")

    ustreznost = int(input("Ustreznost: "))
    berljivost = int(input("Berljivost: "))
    enakost = int(input("Enakost: "))

    grade["topk"] = {"ustreznost": ustreznost, "berljivost": berljivost, "enakost": enakost}

    grades[str(i)] = grade

    quit_ = input("q to quit, enter to continue:")
    if quit_.startswith("q"):
        break

    print("\n\n")

with open(grades_file, "w", encoding="utf-8") as f:
    json.dump(grades, f, indent=4, ensure_ascii=False)

    




    

