import json, os, sys
from tqdm import tqdm

from argparse import ArgumentParser


def parse_data(input_path, output_path):

    if os.path.exists(output_path):
        print(f"File already exists, skipping generation: {output_path}")
        return 

    with open(input_path) as infile:
        hop1 = json.load(infile)

    converted_json = {}
    converted_json['version'] = '0'

    converted_data = []
    for entry in tqdm(hop1):
        paragraphs = [{'context': entry['context'],
                       'qas':[{'answers': [{'answer_start': entry['label_offsets'][0], 'text': entry['label']} for _ in range(3)],
                               'question': entry['question'],
                               'id': entry['_id']}]}]
        data = {"title": "", "paragraphs": paragraphs}
        converted_data.append(data)

    converted_json['data'] = converted_data

    with open(output_path, 'w') as outfile:
        json.dump(converted_json, outfile)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_path', required=True, help="hotpot_train_v1.1.json")
    parser.add_argument('--output_path', required=True, help="hotpot_train_hop1.json")

    args = parser.parse_args()
    parse_data(args.input_path, args.output_path)

    
