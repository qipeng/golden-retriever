# sample usage: python -m scripts.eval_drqa hotpot_hop1_squad_dev-768-50-v3.preds hotpot_dev_fullwiki_v1.json

from collections import Counter
import json, time, re, sys
from tqdm import tqdm
import pandas as pd
import numpy as np
from search.search import *

def main(pred_filename, original_filename):
    batch_size = 200
    Ns = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]
    max_n = max(Ns)


    with open(pred_filename) as f:
        data = json.load(f)


    with open(original_filename) as f:
        original_data = json.load(f)

    reconstructed_data = []
    for idx, entry in enumerate(original_data):
        id = entry['_id']
        query = data[id][0][0]
        gold = set(y[0] for y in entry['supporting_facts'])
        reconstructed_data.append((query, gold))

    batches = [reconstructed_data[b*batch_size:min((b+1)*batch_size, len(data))]
                for b in range((len(data) + batch_size - 1) // batch_size)]




    para1 = Counter()
    para2 = Counter()
    processed = 0
    for batch in tqdm(batches):
        queries = [x[0] for x in batch]
        res = bulk_text_query(queries, topn=max_n, lazy=True)
        # set lazy to true because we don't really care about the json object here
        for r, d in zip(res,batch):
            para1_found = False
            para2_found = False
            for i, para in enumerate(r):
                if para['title'] in d[1]:
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

    print(processed)

    for n in Ns:
        c1 = sum(para1[k] for k in range(n))
        c2 = sum(para2[k] for k in range(n))

        print("Hits@{:2d}: {:.2f}\tP1@{:2d}: {:.2f}\tP2@{:2d}: {:.2f}".format(
            n, 100 * (c1+c2) / 2 / len(data), n, 100 * c1 / len(data), n, 100 * c2 / len(data)))

if __name__ == "__main__":
    pred_filename = sys.argv[1]
    original_filename = sys.argv[2]
    main(pred_filename, original_filename)
