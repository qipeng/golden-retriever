"""
Given:
    hop1 and hop2 files with ES results

Outputs:
    single file containing questions, hop1 and hop2 ES results merged
    use --num_each to control how many contexts are taken from each file
"""

import argparse

from utils.io import load_json_file, write_json_file

def main(args):
    hop1_data = load_json_file(args.hop1_file)
    hop2_data = load_json_file(args.hop2_file)

    out_data = []
    for hop1, hop2 in zip(hop1_data, hop2_data):
        # We're assuming that the hop1 and hop2 files are sorted in the same
        # order. If this doesn't hold, then we would just make a map
        # {id -> entry} for one file.
        assert hop1['_id'] == hop2['_id']

        entry = {}
        entry['_id'] = hop1['_id']
        entry['question'] = hop1['question']
        if args.include_queries:
            entry['hop1_query'] = hop1['query']
            entry['hop2_query'] = hop2['query']

        entry['context'] = []
        all_titles = set()
        for doc in hop1['json_context'][:args.num_each] + hop2['json_context'][:args.num_each]:
            if doc[0] not in all_titles:
                entry['context'].append(doc)
                all_titles.add(doc[0])

        out_data.append(entry)

    write_json_file(out_data, args.out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge hop1 and hop2 results.')
    parser.add_argument('hop1_file')
    parser.add_argument('hop2_file')
    parser.add_argument('out_file', help='filename to write data out to')
    parser.add_argument('--include_queries', action='store_true')
    parser.add_argument('--num_each', default=5,
                        help='number of contexts to take from each hop',
                        type=int)
    args = parser.parse_args()
    main(args)

