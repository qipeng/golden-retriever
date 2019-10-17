"""Preprocess the hop2 input to convert it to the SQuAD format."""

import json as json
import os.path
import argparse
import glob


def parse_data(INPUT_DIR, INPUT_FILE, query1_file=None):
    DATA_PATH = INPUT_DIR + '/' + INPUT_FILE
    print ("Processing", DATA_PATH)
    data = json.load(open(DATA_PATH, 'r'))
    if query1_file:
        # dictionary of key value pairs, example:
        # '5abf04ae5542993fe9a41dbf': [['Ndebele music', 0.6314478516578674]]
        hop1 = json.load(open(INPUT_DIR + '/' + query1_file, 'r'))

    rows = []
    SHUFFLE = False
    for d in data:
        row = {}
        row['title'] = ''
        if query1_file:
            query1 = hop1[d['_id']][0][0]
            row['query1'] = query1 #TODO
        paragraph = {}
        paragraph['context'] = d['context']
        qas = {}
        qas['question'] = d['question']

        # For test set evaluation, we don't have labels
        # Instead we just use (0, "")
        if 'label_offsets' in d:
            start = d['label_offsets'][0]
            span = d['context'][d['label_offsets'][0]:d['label_offsets'][1]]
        else:
            start = 0
            span = ''

        qas['answers'] = [{'answer_start': start, 'text': span}]
        qas['id'] = d['_id']
        paragraph['qas'] = [qas]
        row['paragraphs'] = [paragraph]
        rows.append(row)
        
        if query1_file:
            OUTPUT_FILE = '/SQuAD_query1_' + INPUT_FILE
        else:
            OUTPUT_FILE = '/SQuAD_' + INPUT_FILE

    with open(INPUT_DIR + OUTPUT_FILE, 'w') as outfile:
        json.dump({'data': rows}, outfile)
    
    print ("Done processing. Output to", INPUT_DIR + OUTPUT_FILE)

if __name__ == "__main__":
    # Example usage: python -m scripts.SQuADify_label2 /u/scr/veralin/DrQA/data/datasets hotpot_hop2_dev_v6.json
    # Example usage with query1 file: python -m scripts.SQuADify_label2 /u/scr/veralin/DrQA/data/datasets hotpot_hop2_dev_v7.json --query1_file hotpot_hop1_squad_dev-768-50-v8.preds
    parser = argparse.ArgumentParser(description='Convert Label2_v4 to SQuAD format')
    parser.add_argument('input_dir', help='The input directory ')
    parser.add_argument('input_file', help='The input json file')
    parser.add_argument('--query1_file', help='Include query1 in the question field.')

    args = parser.parse_args()

    parse_data(args.input_dir, args.input_file, query1_file=args.query1_file)
