import copy
import random

def ir_candidates(ir_result, retain=5, force_retain_target=True):
    _id = ir_result['_id']

    target = ir_result['target_para']

    retained = ir_result['ir_result'][:retain]

    if force_retain_target and target['title'] not in [x['title'] for x in retained]:
        retained = retained[:-1] + [target]

    return (_id, [(x['title'], x['text']) for x in retained])

def merge_and_shuffle(orig_datum, hop1_candidates, hop2_candidates):
    # we always process examples in order so alignment should be preserved for free
    assert orig_datum['_id'] == hop1_candidates[0] == hop2_candidates[0]

    all_candidates = []
    all_titles = set()
    for doc in hop1_candidates[1] + hop2_candidates[1]:
        if doc[0] not in all_titles:
            # deduplicate if we can
            all_candidates.append(doc)
            all_titles.add(doc[0])

    random.shuffle(all_candidates)

    res = copy.copy(orig_datum)

    res['context'] = all_candidates

    return res

if __name__ == "__main__":
    import argparse
    import json
    from time import time
    from datetime import timedelta
    from joblib import Parallel, delayed

    RETAIN_HOP1 = 5
    RETAIN_HOP2 = 5

    parser = argparse.ArgumentParser()

    parser.add_argument('split', choices=['train', 'dev-distractor', 'dev-fullwiki'])

    args = parser.parse_args()

    if args.split == 'train':
        input_data = 'data/hotpotqa/hotpot_train_v1.1.json'
        hop1_ir_result = 'data/hop1/hotpot_hop1_train_ir_result.json'
        hop2_ir_result = 'data/hop2/hotpot_hop2_train_ir_result.json'
        output_data = 'data/hotpotqa/hotpot_train_hops.json'
        force_retain = True
    elif args.split == 'dev-distractor':
        input_data = 'data/hotpotqa/hotpot_dev_distractor_v1.json'
        hop1_ir_result = 'data/hop1/hotpot_hop1_dev_ir_result.json'
        hop2_ir_result = 'data/hop2/hotpot_hop2_dev_ir_result.json'
        output_data = 'data/hotpotqa/hotpot_dev_distractor_hops.json'
        force_retain = True
    else:
        input_data = 'data/hotpotqa/hotpot_dev_distractor_v1.json'
        hop1_ir_result = 'data/hop1/hotpot_hop1_dev_ir_result.json'
        hop2_ir_result = 'data/hop2/hotpot_hop2_dev_ir_result.json'
        output_data = 'data/hotpotqa/hotpot_dev_fullwiki_hops.json'
        force_retain = False

    print('Loading HotpotQA training input... ', end="", flush=True)
    t0 = time()
    with open(input_data) as f:
        orig_data = json.load(f)
    print('Done. (took {})'.format(timedelta(seconds=int(time()-t0))), flush=True)

    print('Loading Hop 1 IR result... ', end="", flush=True)
    t0 = time()
    with open(hop1_ir_result) as f:
        hop1 = json.load(f)
    print('Done. (took {})'.format(timedelta(seconds=int(time()-t0))), flush=True)

    hop1_candidates = Parallel(n_jobs=32, verbose=10)(delayed(ir_candidates)(x, retain=RETAIN_HOP1, force_retain_target=force_retain) for x in hop1)
    del hop1

    print('Loading Hop 2 IR result... ', end="", flush=True)
    t0 = time()
    with open(hop2_ir_result) as f:
        hop2 = json.load(f)
    print('Done. (took {})'.format(timedelta(seconds=int(time()-t0))), flush=True)

    hop2_candidates = Parallel(n_jobs=32, verbose=10)(delayed(ir_candidates)(x, retain=RETAIN_HOP2, force_retain_target=force_retain) for x in hop2)
    del hop2

    final = Parallel(n_jobs=32, verbose=10)(delayed(merge_and_shuffle)(x, y, z) for x, y, z in zip(orig_data, hop1_candidates, hop2_candidates))

    print('Saving data file... ', end="", flush=True)
    t0 = time()
    with open(output_data, 'w') as f:
        json.dump(final, f)
    print('Done. (took {})'.format(timedelta(seconds=int(time()-t0))), flush=True)
