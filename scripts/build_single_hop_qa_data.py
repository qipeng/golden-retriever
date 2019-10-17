from collections import Counter
from copy import copy
import json
from tqdm import tqdm

from search.search import bulk_text_query
from utils.general import chunks

def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('split', choices=['train', 'dev'])

    args = parser.parse_args()

    if args.split == 'train':
        filename = 'data/hotpotqa/hotpot_train_v1.1.json'
        outputname = 'data/hotpotqa/hotpot_train_single_hop.json'
    else:
        filename = 'data/hotpotqa/hotpot_dev_fullwiki_v1.json'
        outputname = 'data/hotpotqa/hotpot_dev_single_hop.json'
    batch_size = 64

    with open(filename) as f:
        data = json.load(f)

    outputdata = []
    processed = 0
    for batch in tqdm(chunks(data, batch_size), total=(len(data) + batch_size - 1) // batch_size):
        queries = [x['question'] for x in batch]
        res = bulk_text_query(queries, topn=10, lazy=False)
        for r, d in zip(res, batch):
            d1 = copy(d)
            context = [item['data_object'] for item in r]
            context = [(x['title'], x['text']) for x in context]
            d1['context'] = context
            outputdata.append(d1)

        processed += len(batch)

    with open(outputname, 'w') as f:
        json.dump(outputdata, f)

if __name__ == "__main__":
    main()
