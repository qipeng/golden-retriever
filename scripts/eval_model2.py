from collections import Counter
import json, time, re, sys
from tqdm import tqdm
import pandas as pd
import numpy as np
from search.search import *
import argparse

def evaluate(pred_filename, original_filename):
    batch_size = 200
    Ns = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]
    max_n = max(Ns)

    with open(pred_filename) as f:
        data = json.load(f)

    with open(original_filename) as f:
        original_data = json.load(f)

    if len(original_data) != len(data):
        print("Warning: Data length mismatch")
        print("Label file:", len(original_data))
        print("Preds file:", len(data))


    reconstructed_data = []
    for idx, entry in enumerate(original_data):
        id = entry['_id']
        query = data[id][0][0]
        title1 = entry['title1']
        title2 = entry['title2']
        reconstructed_data.append((query, title1, title2)) 

    batches = [reconstructed_data[b*batch_size:min((b+1)*batch_size, len(data))]
                for b in range((len(original_data) + batch_size - 1) // batch_size)]


    para1 = Counter()
    para2 = Counter()
    for batch in tqdm(batches):
        queries = [x[0] for x in batch]
        res = bulk_text_query(queries, topn=max_n, lazy=True)
        # set lazy to true because we don't really care about the json object here
        for r, d in zip(res, batch):
            query, title1, title2 = d
            para1_found = False
            para2_found = False
            # enumerate search results for current query
            for i, para in enumerate(r):
                if para['title'] == title1 and not para1_found:
                    para1[i] += 1
                    para1_found = True
                elif para['title'] == title2 and not para2_found:
                    para2[i] += 1
                    para2_found = True

    # Print stats
    for n in Ns:
        c1 = sum(para1[k] for k in range(n))
        c2 = sum(para2[k] for k in range(n))

        print("Hits@{:2d}: {:.2f}\tP1@{:2d}: {:.2f}\tP2@{:2d}: {:.2f}".format(
            n, 
            100 * (c1+c2) / 2 / len(reconstructed_data), # Hits@n
            n, 
            100 * c1 / len(reconstructed_data), # P1@n
            n, 
            100 * c2 / len(reconstructed_data) # P2@n
        ))


if __name__ == "__main__":
    # Example usage: python -m scripts.eval_model2 /u/scr/veralin/DrQA/data/datasets/SQuAD_hotpot_hop2_dev_v4-hop2_v2_30e.preds /u/scr/veralin/deep-retriever/data/hop2/hotpot_hop2_dev_v4.json
    parser = argparse.ArgumentParser(description='IR evaluation for model 2 predictions.')
    parser.add_argument('pred_filename', help='The prediction json files ')
    parser.add_argument('original_filename', help='The label json file that contains title1 and title2')

    args = parser.parse_args()

    # pred_filename = sys.argv[1]
    # original_filename = sys.argv[2]

    evaluate(args.pred_filename, args.original_filename) 

