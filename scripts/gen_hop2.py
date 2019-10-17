
import json
from collections import Counter
import sys

from tqdm import tqdm

from search.search import bulk_text_query
from utils.lcs import LCSubStr
from utils.io import load_json_file
from utils.general import chunks, make_context
from utils.corenlp import bulk_tokenize

from scripts.gen_hop1 import CompositeLCS, deduped_bulk_query, STOP_WORDS

def analyze(hop2_results):
    batch_size = 128
    Ns = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]
    max_n = max(Ns)
    p1_hits = Counter()
    p2_hits = Counter()
    processed = 0

    for chunk in tqdm(chunks(hop2_results, batch_size)):

        label2s = [x['label'] for x in chunk]
        es_bulk_results = bulk_text_query(label2s, topn=max_n, lazy=False)

        for i, (entry, es_results) in enumerate(zip(chunk, es_bulk_results)):
            q = entry['question']
            l2 = entry['label']
            t1 = entry['title1']
            p1 = entry['para1']
            t2 = entry['title2']
            p2 = entry['para2']

            # find rank of t1 in es_results
            found_t1 = False
            found_t2 = False
            t2_rank = max_n
            for i, es_entry in enumerate(es_results):
                if es_entry['title'] == t1:
                    p1_hits[i] += 1
                    found_t1 = True
                if es_entry['title'] == t2:
                    p2_hits[i] += 1
                    t2_rank = i
                    found_t2 = True
            if not found_t1:
                p1_hits[max_n] += 1
            if not found_t2:
                p2_hits[max_n] += 1

            print_cols = [q, l2, t1, p1, t2, p2, str(t2_rank + 1)]
            #print('\t'.join(print_cols))
            processed += 1

    for n in Ns:
        c1 = sum(p1_hits[k] for k in range(n))
        c2 = sum(p2_hits[k] for k in range(n))

        print("Hits@{:2d}: {:.2f}\tP1@{:2d}: {:.2f}\tP2@{:2d}: {:.2f}".format(
            n, 100 * (c1+c2) / 2 / processed, n, 100 * c1 / processed, n, 100 * c2 / processed))

def main():
    import argparse

    HOP1_TO_KEEP = 5
    IR_RESULTS_TO_RETAIN = 10
    max_n = 50
    batch_size = 64

    parser = argparse.ArgumentParser()

    parser.add_argument('split', choices=['train', 'dev'])
    parser.add_argument('--analyze', action='store_true')

    args = parser.parse_args()

    if args.split == 'train':
        data_file = 'data/hotpotqa/hotpot_train_v1.1.json'
        labels_file = 'data/hop1/hotpot_hop1_train.json'
        ir_file = 'data/hop1/hotpot_hop1_train_ir_result.json'
        output_file = 'data/hop2/hotpot_hop2_train.json'
        output_ir_file = 'data/hop2/hotpot_hop2_train_ir_result.json'
    else:
        data_file = 'data/hotpotqa/hotpot_dev_distractor_v1.json'
        labels_file = 'data/hop1/hotpot_hop1_dev.json'
        ir_file = 'data/hop1/hotpot_hop1_dev_ir_result.json'
        output_file = 'data/hop2/hotpot_hop2_dev.json'
        output_ir_file = 'data/hop2/hotpot_hop2_dev_ir_result.json'

    # make a map from id to each entry in the data so that we
    # can join with the generated label files
    id_to_datum = {}
    data = load_json_file(data_file)
    for datum in data:
        id_to_datum[datum['_id']] = datum

    # same, map from id to ir entry
    id_to_ir_entry = {}
    ir_data = load_json_file(ir_file)
    for entry in ir_data:
        id_to_ir_entry[entry['_id']] = entry

    hop1_labels= load_json_file(labels_file)

    hop2_results = []
    hop2_ir_results = []
    candidates_per_example = 0
    for batch in tqdm(chunks(hop1_labels, batch_size), total=(len(hop1_labels) + batch_size - 1)//batch_size):
        queries = []
        processed_batch = []
        for entry in batch:
            _id = entry['_id']
            target_para = entry['target_para']

            assert target_para is not None

            title1 = target_para['title']
            para1 = ''.join(target_para['text'])
            question = entry['question']

            orig_datum = id_to_datum[_id]
            supp_facts = set([f[0] for f in orig_datum['supporting_facts']])
            assert len(supp_facts) == 2, supp_facts
            assert title1 in supp_facts
            supp_facts.remove(title1)
            title2 = supp_facts.pop()

            para2_matches = [
                para for title, para in orig_datum['context']
                if title == title2
            ]
            assert len(para2_matches) == 1, orig_datum
            para2 = ''.join(para2_matches[0])
            para2_list = para2_matches[0]

            # join in hop1 ir results
            ir_entry = id_to_ir_entry[_id]

            if title1 in [x['title'] for x in ir_entry['ir_result'][:HOP1_TO_KEEP]]:
                ir_context = ir_entry['ir_result'][:HOP1_TO_KEEP]
            else:
                ir_context = ir_entry['ir_result'][:(HOP1_TO_KEEP-1)] + [{'title': title1, 'text': target_para['text']}]

            hop1_context = make_context(question, ir_context)

            tokenized, offsets = bulk_tokenize(
                    [hop1_context, title2, para2],
                    return_offsets=True
            )
            token_hop1_context = tokenized[0]
            token_title2 = tokenized[1]
            token_para2 = tokenized[2]

            candidates = CompositeLCS(
                    hop1_context,
                    token_hop1_context,
                    token_title2,
                    token_para2,
                    offsets[0],
            )

            if candidates_per_example == 0:
                candidates_per_example = len(candidates)
            assert len(candidates) == candidates_per_example
            queries.extend([x[0] for x in candidates])

            processed_batch.append([_id, entry, question, candidates, target_para, title1, para1, title2, para2, para2_list, hop1_context])

        res = deduped_bulk_query(queries, topn=max_n, lazy=False)

        for i, (_id, entry, question, candidates, target_para, title1, para1, title2, para2, para2_list, hop1_context) in enumerate(processed_batch):
            def process_result_item(query_offsets, item, item_idx):
                rank = min([i for i, para in enumerate(item) if para['title'] == title2] + [max_n])
                target_para = item[rank]['data_object'] if rank < max_n else None
                if target_para is None:
                    target_para = {'title': title2, 'text': para2_list}
                query, offsets = query_offsets
                splitted = [x for x in query.split() if len(x)]
                token_len = len(splitted)
                if len(splitted) == 0:
                    upper_case_len = 0
                else:
                    upper_case_len = sum((not x[0].islower()) or x in ['in', 'the', 'of', 'by', 'a', 'an', 'on', 'to', 'is'] for x in splitted) if len(splitted) <= 5 and (not splitted[0][0].islower()) and (not splitted[-1][0].islower()) and splitted[-1].lower() not in STOP_WORDS else sum(not x[0].islower() for x in splitted)
                return max(4, rank), max(token_len, 10) + rank, offsets[:2], rank, item_idx, query, target_para, token_len

            res1 = [process_result_item(q1, r1, idx) for idx, q1, r1 in zip(range(len(candidates)), candidates, res[i*candidates_per_example:(i+1)*candidates_per_example])]

            _, _, offsets, rank, res_idx, query, target_para, token_len = list(sorted(res1))[0]

            hop2_ir_results.append({
                '_id': _id,
                'query': query,
                'target_para': target_para,
                'target_rank': rank,
                'ir_result': [x['data_object'] for x in res[i*candidates_per_example+res_idx][:IR_RESULTS_TO_RETAIN]]
            })

            hop2_results.append({
                '_id': _id,
                'question': question,
                'label': query,
                'context': hop1_context,
                'label_offsets': offsets,
                'hop1_label': entry['label'],
                'hop1_offsets': entry['label_offsets'],
                'title1': title1,
                'para1': para1,
                'title2': title2,
                'para2': para2,
            })

    print('Dumping Hop 2 labels to file... ', end="", flush=True)
    with open(output_file, 'w') as f:
        json.dump(hop2_results, f)
    print('Done.', flush=True)

    print('Dumping IR result to file... ', end="", flush=True)
    with open(output_ir_file, 'w') as f:
        json.dump(hop2_ir_results, f)
    print('Done.', flush=True)

    if args.analyze:
        analyze(hop2_results)

    print('Done!', file=sys.stderr)

if __name__ == "__main__":
    main()
