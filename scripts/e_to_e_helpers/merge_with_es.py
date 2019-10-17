"""
Query ES and merge results with original hotpot data.

Input:
    - query file
    - hotpotqa data
    - output filename
    - whether this is for hop1 or hop2

Outputs:
    - json file containing a list of:
        {'context', 'question', '_id', 'query', 'json_context'}
        context -- the concatentation of the top n paragraphs for the given query
            to ES.
        json_context -- same as context, but in json structure same as original
            hotpot data.
        question, _id -- identical to those from the original HotPotQA data
"""

import argparse
from tqdm import tqdm
from search.search import bulk_text_query
from utils.io import load_json_file, write_json_file
from utils.general import chunks, make_context

def main(query_file, question_file, out_file, top_n):
    query_data = load_json_file(query_file)
    question_data = load_json_file(question_file)

    out_data = []

    for chunk in tqdm(list(chunks(question_data, 100))):
        queries = []
        for datum in chunk:
            _id = datum['_id']
            queries.append(query_data[_id] if isinstance(query_data[_id], str) else query_data[_id][0][0])

        es_results = bulk_text_query(queries, topn=top_n, lazy=False)
        for es_result, datum in zip(es_results, chunk):
            _id = datum['_id']
            question = datum['question']
            query = query_data[_id] if isinstance(query_data[_id], str) else query_data[_id][0][0]
            context = make_context(question, es_result)
            json_context = [
                [p['title'], p['data_object']['text']]
                for p in es_result
            ]

            out_data.append({
                '_id': _id,
                'question': question,
                'context': context,
                'query': query,
                'json_context': json_context
            })

        write_json_file(out_data, out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Query ES and merge results with original hotpot data.')
    parser.add_argument('query_file', help='.preds file containing ES queries ')
    parser.add_argument('question_file', help='.json file containing original questions and ids')
    parser.add_argument('out_file', help='filename to write data out to')
    parser.add_argument('--top_n', default=5,
            help='number of docs to return from  ES',
            type=int)
    args = parser.parse_args()

    main(args.query_file, args.question_file, args.out_file, args.top_n)

