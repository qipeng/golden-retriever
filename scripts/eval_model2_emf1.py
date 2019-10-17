import ujson as json
import re
import string
from collections import Counter
import argparse


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def evaluate(prediction_file, gold_file):
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)['data']

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
    for dp in gold:
        cur_id = dp['paragraphs'][0]['qas'][0]['id']
        if cur_id not in prediction:
            print('missing answer {}'.format(cur_id))
        else:
            cur_pred = prediction[cur_id][0][0]
            cur_label = dp['paragraphs'][0]['qas'][0]['answers'][0]['text']
            em, prec, recall = update_answer(metrics, cur_pred, cur_label)
    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    print(metrics)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model 2')
    parser.add_argument('prediction_file', help='The prediction file')
    parser.add_argument('gold_file', help='The gold file')

    args = parser.parse_args()
    
    evaluate(args.prediction_file, args.gold_file)
    
    

