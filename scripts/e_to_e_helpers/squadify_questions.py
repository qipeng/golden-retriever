"""
Given a list of questions, produces them in SQuAD format for DrQA.

Input file should be in json, a list of objects each of which
must have at least a "question" and an "_id".
"""

from argparse import ArgumentParser
from tqdm import tqdm

from utils.io import write_json_file, load_json_file

def main(question_file, out_file):
    data = load_json_file(question_file)

    rows = []
    for entry in data:
        assert 'question' in entry, 'every entry must have a question'
        assert '_id' in entry, 'every entry must have an _id'
        row = {
            'title': '',
            'paragraphs': [{
                'context': entry['question'],
                'qas': [{
                    'question': entry['question'],
                    'id': entry['_id'],
                    'answers': [{'answer_start': 0, 'text': ''}]
                }]
            }]
        }
        rows.append(row)

    write_json_file({'data': rows}, out_file)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('question_file',
        help="json file containing a list of questions and IDs")
    parser.add_argument('out_file',
        help="File to output SQuAD-formatted questions to")

    args = parser.parse_args()
    main(args.question_file, args.out_file)
