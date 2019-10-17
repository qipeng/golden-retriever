from collections import Counter
import json
import sys
from tqdm import tqdm

with open(sys.argv[1]) as f:
    qa_input = json.load(f)

with open(sys.argv[2]) as f:
    eval_file = json.load(f)

recall = 0
total = 0
foundall = 0
foundall_total = 0
foundone = 0
foundone_total = 0

bridge_recall = 0
bridge_total = 0
bridge_foundall = 0
bridge_foundall_total = 0
bridge_foundone = 0
bridge_foundone_total = 0
for d1, d2 in zip(tqdm(qa_input), eval_file):
    assert d1['_id'] == d2['_id']

    c1 = set([x[0] for x in d1['context']])
    c2 = set([x[0] for x in d2['supporting_facts']])

    assert len(c2) == 2

    found = len(c1 & c2)
    recall += found
    total += len(c2)

    foundall += (found == len(c2))
    foundall_total += 1

    foundone += (found >= 1)
    foundone_total += 1

    if d2['type'] != 'comparison':
        found = len(c1 & c2)
        bridge_recall += found
        bridge_total += len(c2)

        bridge_foundall += (found == len(c2))
        bridge_foundall_total += 1

        bridge_foundone += (found >= 1)
        bridge_foundone_total += 1


print('Recall: {:5.2%} ({:5d} / {:5d})'.format(recall / total, recall, total))
print('Found all: {:5.2%} ({:5d} / {:5d})'.format(foundall / foundall_total, foundall, foundall_total))
print('Found one: {:5.2%} ({:5d} / {:5d})'.format(foundone / foundone_total, foundone, foundone_total))
print()
print('Bridge-only recall: {:5.2%} ({:5d} / {:5d})'.format(bridge_recall / bridge_total, bridge_recall, bridge_total))
print('Bridge-only found all: {:5.2%} ({:5d} / {:5d})'.format(bridge_foundall / bridge_foundall_total, bridge_foundall, bridge_foundall_total))
print('Bridge-only found one: {:5.2%} ({:5d} / {:5d})'.format(bridge_foundone / bridge_foundone_total, bridge_foundone, bridge_foundone_total))
print()
comparison_recall = recall - bridge_recall
comparison_total = total - bridge_total
comparison_foundall = foundall - bridge_foundall
comparison_foundall_total = foundall_total - bridge_foundall_total
comparison_foundone = foundone - bridge_foundone
comparison_foundone_total = foundone_total - bridge_foundone_total
print('Comparison-only recall: {:5.2%} ({:5d} / {:5d})'.format(comparison_recall / comparison_total, comparison_recall, comparison_total))
print('Comparison-only found all: {:5.2%} ({:5d} / {:5d})'.format(comparison_foundall / comparison_foundall_total, comparison_foundall, comparison_foundall_total))
print('Comparison-only found one: {:5.2%} ({:5d} / {:5d})'.format(comparison_foundone / comparison_foundone_total, comparison_foundone, comparison_foundone_total))
