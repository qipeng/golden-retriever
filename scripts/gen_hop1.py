from collections import Counter
from editdistance import eval as editdistance
import json
from multiprocessing import Pool
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import os
from pprint import pprint
import re
from tqdm import tqdm

from search.search import bulk_text_query, core_title_filter
from utils.constant import SEP
from utils.corenlp import bulk_tokenize
from utils.lcs import LCSubStr, LCS

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS2 = set(stopwords.words('english') + [',', '.', ';', '?', '"', '\'', '(', ')', '&', '?'])

EXPANSION = {r'(movie|film)s?': 'film',
        r'novels?': 'novel',
        r'bands?': 'band',
        r'books?': 'book',
        r'magazines?': 'magazine',
        r'albums?': 'album',
        r'operas?': 'opera',
        r'episodes?': 'episode',
        r'series': 'series',
        r'board\s+games?': 'board game',
        r'director': 'film TV series',
        r'publish': 'book novel magazine',
        r'(cocktail|drink)s': 'cocktail alcohol'}

def expand_query(question, query):
    return query

    for pattern in EXPANSION:
        if re.search(pattern, question):
            query += " {}".format(EXPANSION[pattern])

    return query

stemmer = PorterStemmer()

def _filter_stopwords(text):
    res = [(x, i) for i, x in enumerate(text) if not x in STOP_WORDS]
    if len(res) == 0:
        return [], []
        #res = [(x, i) for i, x in enumerate(text)]
    return map(list, zip(*res))

def _filter_stopwords2(text):
    res = [(x, i) for i, x in enumerate(text) if not x in STOP_WORDS2]
    return map(list, zip(*res))

def CompositeLCS(context_orig, context, title, para, ctx_offsets, TIE_BREAKER=0):
    """
    inputs:
    - context_orig: str, raw text of the "context" that our query generator would see at this stage
    - context: list, list of tokens in the context
    - title: title of the target gold paragraph
    - para: paragraph text of the gold paragraph
    - ctx_offsets: the character offsets of the words in the context, as returned by corenlp
    - TIE_BREAKER: deprecated, used to break ties between matches between LCS and LCSubStr matches

    returns:
    - a list of [question, (start_char_offset, end_char_offset, start_token_offset, end_token_offset)]
      of candidate queries to evaluate against elasticsearch
    """


    q0, q, t1, c1 = context_orig, context, title, para
    q_, qidx = _filter_stopwords2(q)
    t1_, _ = _filter_stopwords(t1)
    c1_, c1idx = _filter_stopwords(c1)

    q_ = [x.lower() for x in q_]
    t1_ = [x.lower() for x in t1_]
    c1_ = [x.lower() for x in c1_]

    leading_whitespaces = len(q0) - len(q0.lstrip())

    def map_indices(idx1):
        """
        map matched token offsets back to character offsets and original token offsets
        """
        if idx1[0] < 0 or idx1[0] == idx1[1]:
            idx1 = 0, len(qidx)
        return (ctx_offsets[qidx[idx1[0]]][0] + leading_whitespaces,
                ctx_offsets[qidx[idx1[1]-1]][1] + leading_whitespaces,
                qidx[idx1[0]],
                qidx[idx1[1]-1]+1)

    def map_indices0(idx1):
        """
        map matched token offsets back to character offsets and original token offsets
        """
        if idx1[0] < 0 or idx1[0] == idx1[1]:
            idx1 = 0, len(q)
        return (ctx_offsets[idx1[0]][0] + leading_whitespaces,
                ctx_offsets[idx1[1]-1][1] + leading_whitespaces,
                idx1[0],
                idx1[1])

    l1, _, idx1 = LCS(q_, c1_)
    l1t, _, idx1t = LCS(q_, t1_)
    l1_substr, _, idx1_substr = LCSubStr(q_, c1_)
    l1t_substr, _, idx1t_substr = LCSubStr(q_, t1_)

    q_0 = [x.lower() for x in q]
    t1_0 = [x.lower() for x in t1]
    l1t_substr0, _, idx1t_substr0 = LCSubStr(q_0, t1_0)

    def typo_aware_in(x, tgt_set, min_len, tolerance):
        for c in tgt_set:
            if len(c) >= min_len and editdistance(x, c) <= tolerance:
                return True

        return False

    def find_overlap(q, c, replacement, qidx):
        cset = set(c)
        overlap = []
        overlapping = False
        last = -1
        for i, x in enumerate(q):
            if x in cset or typo_aware_in(x, cset, 3, 1):
                if x in ['"']:
                    if last >= 0:
                        overlap.append((last, i))
                        last = -1
                    overlap.append((i, i+1))
                    overlapping = False
                    continue
                if not overlapping:
                    last = i
                    overlapping = True
            else:
                if last >= 0:
                    overlap.append((last, i))
                    last = -1
                overlapping = False

        if overlapping:
            overlap.append((last, len(q)))

        target_count = 4
        if len(overlap) > 0:
            cands = [(-sum(x[1]-x[0] for x in overlap[i:j+1])/(overlap[j][1]-overlap[i][0]), overlap[i][0], overlap[j][1]) if '<t>' not in q[overlap[i][0]:overlap[j][1]] and '</t>' not in q[overlap[i][0]:overlap[j][1]] else (1e10, overlap[i][0], overlap[j][1]) for i in range(len(overlap)) for j in range(i, len(overlap))]
            cands = [(x[1], x[2]) for x in list(sorted(cands))[:target_count]]
            cands += [replacement] * (target_count - len(cands))
            return cands
        else:
            return [replacement] * target_count

    idx2t = find_overlap(q_, t1_, idx1t, qidx)

    idx2 = find_overlap(q_, c1_, idx1, qidx)

    cand_offsets = [map_indices(idx) for idx in [idx1, idx1t, idx1_substr, idx1t_substr] + idx2t + idx2]

    cand_offsets += [map_indices0(idx) for idx in [idx1t_substr0]]

    return [(q0[st:en], (st, en, st2, en2)) for st, en, st2, en2 in cand_offsets]

def generate_single_hop1_query(data, TIE_BREAKER=0):
    context = dict(data['context'])
    supporting = sorted(set([x[0] for x in data['supporting_facts']]))

    q0 = data['question'].strip()
    c1 = [supporting[0], ''.join(context[supporting[0]])]
    c2 = [supporting[1], ''.join(context[supporting[1]])]

    to_tok = [q0] + c1 + c2
    tokenized, offsets = bulk_tokenize(to_tok, return_offsets=True)
    q = tokenized[0]
    t1, c1 = tokenized[1:3]
    t2, c2 = tokenized[3:5]

    cands1 = CompositeLCS(data['question'], q, t1, c1, offsets[0], TIE_BREAKER=TIE_BREAKER)
    cands2 = CompositeLCS(data['question'], q, t2, c2, offsets[0], TIE_BREAKER=TIE_BREAKER)

    return cands1, cands2

def generate_hop1_queries(data, TIE_BREAKER=0):
    return [generate_single_hop1_query(datum, TIE_BREAKER=TIE_BREAKER) for datum in data]

def deduped_bulk_query(queries1, topn=10, lazy=True):
    # consolidate queries to remove redundancy
    queries2 = []
    queries2_dict = dict()
    mapped_idx = []
    for q in queries1:
        if q not in queries2_dict:
            queries2_dict[q] = len(queries2)
            queries2.append(q)
        mapped_idx.append(queries2_dict[q])

    res1 = bulk_text_query(queries2, topn=topn, lazy=lazy)

    # map queries back
    res = [res1[idx] for idx in mapped_idx]

    return res

def main():
    import argparse

    IR_RESULTS_TO_RETAIN = 10

    sanitycheck = dict()
    if os.path.exists('data/hop1/sanitycheck.tsv'):
        with open('data/hop1/sanitycheck.tsv') as f:
            for line in f:
                line = line.rstrip().split('\t')
                sanitycheck[line[0]] = line[1]

    parser = argparse.ArgumentParser()

    parser.add_argument('split', choices=['train', 'dev'])
    parser.add_argument('--analysis', action='store_true')

    args = parser.parse_args()

    if args.split == 'train':
        filename = 'data/hotpotqa/hotpot_train_v1.1.json'
        labels_file = 'data/hop1/hotpot_hop1_train.json'
        ir_result_file = 'data/hop1/hotpot_hop1_train_ir_result.json'
    else:
        filename = 'data/hotpotqa/hotpot_dev_distractor_v1.json'
        labels_file = 'data/hop1/hotpot_hop1_dev.json'
        ir_result_file = 'data/hop1/hotpot_hop1_dev_ir_result.json'

    batch_size = 64
    Ns = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]
    max_n = max(Ns)

    with open(filename) as f:
        data = json.load(f)

    batches = [data[b*batch_size:min((b+1)*batch_size, len(data))]
            for b in range((len(data) + batch_size - 1) // batch_size)]

    para1 = Counter()
    para2 = Counter()
    processed = 0

    ir_result = []
    hop1_labels = []

    f_analysis = open('hop1_analysis_ge5.tsv', 'w') if args.analysis else None
    candidates_per_paragraph = 0

    pool = Pool(8)
    all_queries = list(tqdm(pool.imap(generate_hop1_queries, batches), total=len(batches)))
    for batch, queries in tqdm(zip(batches, all_queries), total=len(batches)):
        #queries = generate_hop1_queries(batch)
        if candidates_per_paragraph == 0:
            candidates_per_paragraph = len(queries[0][0])
            print('Candidates per paragraph evaluated: {}'.format(candidates_per_paragraph))
        assert all([len(y) == candidates_per_paragraph for x in queries for y in x])

        queries1 = [expand_query(d['question'], z[0]) for x, d in zip(queries, batch) for y in x for z in y]

        res = deduped_bulk_query(queries1, topn=max_n, lazy=True)

        for j, d, q in zip(range(len(batch)), batch, queries):
            supporting = sorted(set(x[0] for x in d['supporting_facts'])) # paragraph titles
            supporting1 = sorted(set(core_title_filter(x[0]) for x in d['supporting_facts'])) # paragraph titles
            ctx = dict(d['context'])

            def process_result_item(query_offsets, item, orig_target, item_idx):
                rank = min([i for i, para in enumerate(item) if para['title'] in supporting] + [max_n])
                target_para = json.loads(item[rank]['data_object']) if rank < max_n else None
                if target_para is None:
                    target_para = {'title': orig_target, 'text': ctx[orig_target]}
                query, offsets = query_offsets
                splitted = query.split()
                token_len = len(splitted)
                ques_len = len(d['question'].split())
                upper_case_len = sum((not x[0].islower()) or x in ['in', 'the', 'of', 'by', 'a', 'an', 'on', 'to', 'is'] for x in splitted) if len(splitted) <= 5 and (not splitted[0][0].islower()) and (not splitted[-1][0].islower()) and splitted[-1].lower() not in STOP_WORDS else sum(not x[0].islower() for x in splitted)
                return max(4, rank), max(token_len, min(10, ques_len * .6)) + offsets[2] * .1 + rank + max(1, sum(title in query for title in supporting1)), offsets[:2], rank, item_idx, query, target_para, token_len

            res1 = [process_result_item(q1, r1, supporting[0], idx) for idx, q1, r1 in zip(range(len(q[0])), q[0], res[j*2*candidates_per_paragraph:(j*2+1)*candidates_per_paragraph])]
            res1 += [process_result_item(q1, r1, supporting[1], idx+len(q[0])) for idx, q1, r1 in zip(range(len(q[1])), q[1], res[(j*2+1)*candidates_per_paragraph:(j*2+2)*candidates_per_paragraph])]

            _, _, offsets, rank, res_idx, query, target_para, token_len = list(sorted(res1))[0]
            r = res[j*2*candidates_per_paragraph:(j*2+2)*candidates_per_paragraph][res_idx]

            ir_result.append({
                '_id': d['_id'],
                'query': query,
                'target_para': target_para,
                'target_rank': rank,
                'ir_result': [json.loads(x['data_object']) for x in r[:IR_RESULTS_TO_RETAIN]]
            })

            hop1_labels.append({
                '_id': d['_id'],
                'question': d['question'],
                'context': d['question'],
                'label': query,
                'label_offsets': offsets,
                'target_para': target_para,
                'target_rank': rank,
            })

            para1_found = False
            para2_found = False
            for i, para in enumerate(r):
                if para['title'] in supporting:
                    if not para1_found:
                        para1[i] += 1
                        para1_found = True
                    else:
                        assert not para2_found
                        para2[i] += 1
                        para2_found = True

            if not para1_found:
                para1[max_n] += 1
            if not para2_found:
                para2[max_n] += 1

        processed += len(batch)

    if f_analysis is not None:
        f_analysis.close()

    print('Dumping IR result to file... ', end="", flush=True)
    with open(ir_result_file, 'w') as f:
        json.dump(ir_result, f)
    print('Done.', flush=True)

    print('Dumping Hop 1 labels to file... ', end="", flush=True)
    with open(labels_file, 'w') as f:
        json.dump(hop1_labels, f)
    print('Done.', flush=True)

    for n in Ns:
        c1 = sum(para1[k] for k in range(n))
        c2 = sum(para2[k] for k in range(n))

        print("Hits@{:2d}: {:.2f}\tP1@{:2d}: {:.2f}\tP2@{:2d}: {:.2f}".format(
            n, 100 * (c1+c2) / 2 / processed, n, 100 * c1 / processed, n, 100 * c2 / processed))

if __name__ == '__main__':
    main()
