input_file = "data/mt5-base-rul-pruned_results.json"
output_file = "grades/auto_mt5-base-rul-pruned.json"

import json

from similarity import calculate_bleu_similarity, calculate_cosine_similarity, calculate_meteor_score
from evaluate import load


def get_top(top_k, original):
    min_score = 100000
    min_text = ""
    scores = []

    for i in top_k:
        bleu_score = calculate_bleu_similarity(original, i)

        scores.append(bleu_score)

    sorted_scores = sorted(scores, reverse=True)
    middle = sorted_scores[len(sorted_scores) // 2]

    for i in scores:
        if middle == i:
            return top_k[scores.index(i)]

with open(input_file, "r", encoding="utf8") as f:
    data = json.load(f)

grades = []
bertscore = load("bertscore")

for i in range(len(data)):
    phrases = data[i]

    print(phrases["input"])

    grade = {"greedy": {}, "topk": {}}

    grade["greedy"]["cosine"] = calculate_cosine_similarity(phrases["input"], phrases["greedy"])
    grade["greedy"]["bleu"] = calculate_bleu_similarity(phrases["input"], phrases["greedy"])
    grade["greedy"]["meteor"] = calculate_meteor_score(phrases["input"], phrases["greedy"])
    grade["greedy"]["bertscore"] = bertscore.compute(predictions=[phrases["greedy"]], references=[phrases["input"]], lang="sl")
    print(grade["greedy"]["bertscore"])

    top = get_top(phrases["topk"], phrases["input"])
    grade["topk"]["cosine"] = calculate_cosine_similarity(phrases["input"], top)
    grade["topk"]["bleu"] = calculate_bleu_similarity(phrases["input"], top)
    grade["topk"]["meteor"] = calculate_meteor_score(phrases["input"], top)
    grade["topk"]["bertscore"] = bertscore.compute(predictions=[top], references=[phrases["input"]], lang="sl")

    grades.append(grade)


with open(output_file, "w", encoding="utf8") as f:
    json.dump(grades, f, indent=4)




