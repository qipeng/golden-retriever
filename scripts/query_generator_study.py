import json

dev_file = "data/hotpotqa/hotpot_dev_distractor_v1.json"
hop1_file = "data/hop1/hotpot_hop1_dev_v8.json"
hop2_file = "data/hop2/hotpot_hop2_dev_v7.json"
hop1_pred_file = "hotpotqa_dev_eval/hop1_squadified-32-128-500-v8.preds"
hop2_pred_file = "hotpotqa_dev_eval/SQuAD_hop2_input-hop2_v3_gpu_30e.preds"

with open(dev_file) as f:
    dev_data = json.load(f)

with open(hop1_file) as f:
    hop1_data = json.load(f)

with open(hop2_file) as f:
    hop2_data = json.load(f)

with open(hop1_pred_file) as f:
    hop1_pred = json.load(f)

with open(hop2_pred_file) as f:
    hop2_pred = json.load(f)

for d, h1, h2 in zip(dev_data, hop1_data, hop2_data):
    assert d['_id'] == h1['_id'] == h2['_id']
    id = d['_id']

    support = list(set([x[0] for x in d['supporting_facts']]))
    print('\t'.join([d['question']] + support + [h1['label'], hop1_pred[id][0][0], h2['label'], hop2_pred[id][0][0]]).replace('"', '&quot;'))
