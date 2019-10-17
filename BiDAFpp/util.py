import torch
import numpy as np
import re
from collections import Counter
import string
import pickle
import random
from torch.autograd import Variable
import copy
import ujson as json
import traceback
import bisect

from torch.utils.data import Dataset, DataLoader

IGNORE_INDEX = -100

NUM_OF_PARAGRAPHS = 10
MAX_PARAGRAPH_LEN = 400

RE_D = re.compile('\d')
def has_digit(string):
    return RE_D.search(string)

def prepro(token):
    return token if not has_digit(token) else 'N'

def pad_data(data, sizes, dtype=np.int64, out=None):
    res = np.zeros(sizes, dtype=dtype) if out is None else out
    if len(sizes) == 1:
        res[:min(len(data), sizes[0])] = data[:sizes[0]]
    elif len(sizes) == 2:
        for i, x in enumerate(data):
            if i >= sizes[0]: break
            res[i, :min(len(x), sizes[1])] = data[i][:sizes[1]]
    elif len(sizes) == 3:
        for i, x in enumerate(data):
            if i >= sizes[0]: break
            for j, y in enumerate(x):
                if j >= sizes[1]: break
                res[i, j, :min(len(y), sizes[2])] = data[i][j][:sizes[2]]

    return res#torch.from_numpy(res)

class HotpotDataset(Dataset):
    def __init__(self, buckets):
        self.buckets = buckets
        self.cumlens = []
        for i, b in enumerate(self.buckets):
            last = 0 if i == 0 else self.cumlens[-1]
            self.cumlens.append(last + len(b))

    def __len__(self):
        return self.cumlens[-1]

    def __getitem__(self, i):
        bucket_id = bisect.bisect_right(self.cumlens, i)
        offset = 0 if bucket_id == 0 else self.cumlens[bucket_id-1]
        return self.buckets[bucket_id][i - offset]

class DataIterator(DataLoader):
    def __init__(self, dataset, para_limit, ques_limit, char_limit, sent_limit, **kwargs):
        if kwargs.get('collate_fn', None) is None:
            kwargs['collate_fn'] = self._collate_fn
        if para_limit is not None and ques_limit is not None:
            self.para_limit = para_limit
            self.ques_limit = ques_limit
        else:
            para_limit, ques_limit = 0, 0
            for bucket in buckets:
                for dp in bucket:
                    para_limit = max(para_limit, dp['context_idxs'].size(0))
                    ques_limit = max(ques_limit, dp['ques_idxs'].size(0))
            self.para_limit, self.ques_limit = para_limit, ques_limit

        self.char_limit = char_limit
        self.sent_limit = sent_limit

        super().__init__(dataset, **kwargs)

    def _collate_fn(self, batch_data):
        # Change: changing the dimensions of context_idxs
        batch_size = len(batch_data)
        max_sent_cnt = max(len([y for x in batch_data[i]['start_end_facts'] for y in x]) for i in range(len(batch_data)))

        context_idxs = np.zeros((batch_size, NUM_OF_PARAGRAPHS, MAX_PARAGRAPH_LEN), dtype=np.int64)
        ques_idxs = np.zeros((batch_size, self.ques_limit), dtype=np.int64)
        context_char_idxs = np.zeros((batch_size, NUM_OF_PARAGRAPHS, MAX_PARAGRAPH_LEN, self.char_limit), dtype=np.int64)
        ques_char_idxs = np.zeros((batch_size, self.ques_limit, self.char_limit), dtype=np.int64)
        y1 = np.zeros(batch_size, dtype=np.int64)
        y2 = np.zeros(batch_size, dtype=np.int64)
        q_type = np.zeros(batch_size, dtype=np.int64)
        start_mapping = np.zeros((batch_size, max_sent_cnt, NUM_OF_PARAGRAPHS * MAX_PARAGRAPH_LEN), dtype=np.float32)
        end_mapping = np.zeros((batch_size, max_sent_cnt, NUM_OF_PARAGRAPHS * MAX_PARAGRAPH_LEN), dtype=np.float32)
        all_mapping = np.zeros((batch_size, max_sent_cnt, NUM_OF_PARAGRAPHS * MAX_PARAGRAPH_LEN), dtype=np.float32)
        is_support = np.full((batch_size, max_sent_cnt), IGNORE_INDEX, dtype=np.int64)

        ids = [x['id'] for x in batch_data]

        max_sent_cnt = 0

        for i in range(len(batch_data)):
            pad_data(batch_data[i]['context_idxs'], (NUM_OF_PARAGRAPHS, MAX_PARAGRAPH_LEN), out=context_idxs[i])
            pad_data(batch_data[i]['ques_idxs'], (self.ques_limit,), out=ques_idxs[i])
            pad_data(batch_data[i]['context_char_idxs'], (NUM_OF_PARAGRAPHS, MAX_PARAGRAPH_LEN, self.char_limit), out=context_char_idxs[i])
            pad_data(batch_data[i]['ques_char_idxs'], (self.ques_limit, self.char_limit), out=ques_char_idxs[i])
            if batch_data[i]['y1'] >= 0:
                y1[i] = batch_data[i]['y1']
                y2[i] = batch_data[i]['y2']
                q_type[i] = 0
            elif batch_data[i]['y1'] == -1:
                y1[i] = IGNORE_INDEX
                y2[i] = IGNORE_INDEX
                q_type[i] = 1
            elif batch_data[i]['y1'] == -2:
                y1[i] = IGNORE_INDEX
                y2[i] = IGNORE_INDEX
                q_type[i] = 2
            elif batch_data[i]['y1'] == -3:
                y1[i] = IGNORE_INDEX
                y2[i] = IGNORE_INDEX
                q_type[i] = 3
            else:
                assert False

            for j, (para_id, cur_sp_dp) in enumerate((para_id, s) for para_id, para in enumerate(batch_data[i]['start_end_facts']) for s in para):
                if j >= self.sent_limit: break
                if len(cur_sp_dp) == 3:
                    start, end, is_sp_flag = tuple(cur_sp_dp)
                else:
                    start, end, is_sp_flag, is_gold = tuple(cur_sp_dp)
                start += para_id * MAX_PARAGRAPH_LEN
                end += para_id * MAX_PARAGRAPH_LEN
                if start < end:
                    start_mapping[i, j, start] = 1
                    end_mapping[i, j, end-1] = 1
                    all_mapping[i, j, start:end] = 1
                    is_support[i, j] = int(is_sp_flag)

        input_lengths = (context_idxs > 0).astype(np.int64).sum(2)
        max_q_len = int((ques_idxs > 0).astype(np.int64).sum(1).max())

        context_idxs = torch.from_numpy(context_idxs)
        ques_idxs = torch.from_numpy(ques_idxs[:, :max_q_len])
        context_char_idxs = torch.from_numpy(context_char_idxs)
        ques_char_idxs = torch.from_numpy(ques_char_idxs[:, :max_q_len])
        input_lengths = torch.from_numpy(input_lengths)
        y1 = torch.from_numpy(y1)
        y2 = torch.from_numpy(y2)
        q_type = torch.from_numpy(q_type)
        is_support = torch.from_numpy(is_support)
        start_mapping = torch.from_numpy(start_mapping)
        end_mapping = torch.from_numpy(end_mapping)
        all_mapping = torch.from_numpy(all_mapping)

        return {'context_idxs': context_idxs,
            'ques_idxs': ques_idxs,
            'context_char_idxs': context_char_idxs,
            'ques_char_idxs': ques_char_idxs,
            'context_lens': input_lengths,
            'y1': y1,
            'y2': y2,
            'ids': ids,
            'q_type': q_type,
            'is_support': is_support,
            'start_mapping': start_mapping,
            'end_mapping': end_mapping,
            'all_mapping': all_mapping}

def get_buckets(record_file):
    # datapoints = pickle.load(open(record_file, 'rb'))
    datapoints = torch.load(record_file)
    return [datapoints]

def convert_tokens(eval_file, qa_id, pp1, pp2, p_type):
    answer_dict = {}
    for qid, p1, p2, type in zip(qa_id, pp1, pp2, p_type):
        if type == 0:
            context = eval_file[str(qid)]["context"]
            spans = eval_file[str(qid)]["spans"]
            start_idx = spans[p1][0]
            end_idx = spans[p2][1]
            answer_dict[str(qid)] = context[start_idx: end_idx]
        elif type == 1:
            answer_dict[str(qid)] = 'yes'
        elif type == 2:
            answer_dict[str(qid)] = 'no'
        elif type == 3:
            answer_dict[str(qid)] = 'noanswer'
        else:
            assert False
    return answer_dict

def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answer"]
        prediction = value
        assert len(ground_truths) == 1
        cur_EM = exact_match_score(prediction, ground_truths[0])
        cur_f1, _, _ = f1_score(prediction, ground_truths[0])
        exact_match += cur_EM
        f1 += cur_f1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

# def evaluate(eval_file, answer_dict, full_stats=False):
#     if full_stats:
#         with open('qaid2type.json', 'r') as f:
#             qaid2type = json.load(f)
#         f1_b = exact_match_b = total_b = 0
#         f1_4 = exact_match_4 = total_4 = 0

#         qaid2perf = {}

#     f1 = exact_match = total = 0
#     for key, value in answer_dict.items():
#         total += 1
#         ground_truths = eval_file[key]["answer"]
#         prediction = value
#         cur_EM = metric_max_over_ground_truths(
#             exact_match_score, prediction, ground_truths)
#         # cur_f1 = metric_max_over_ground_truths(f1_score,
#                                             # prediction, ground_truths)
#         assert len(ground_truths) == 1
#         cur_f1, cur_prec, cur_recall = f1_score(prediction, ground_truths[0])
#         exact_match += cur_EM
#         f1 += cur_f1
#         if full_stats and key in qaid2type:
#             if qaid2type[key] == '4':
#                 f1_4 += cur_f1
#                 exact_match_4 += cur_EM
#                 total_4 += 1
#             elif qaid2type[key] == 'b':
#                 f1_b += cur_f1
#                 exact_match_b += cur_EM
#                 total_b += 1
#             else:
#                 assert False

#         if full_stats:
#             qaid2perf[key] = {'em': cur_EM, 'f1': cur_f1, 'pred': prediction,
#                     'prec': cur_prec, 'recall': cur_recall}

#     exact_match = 100.0 * exact_match / total
#     f1 = 100.0 * f1 / total

#     ret = {'exact_match': exact_match, 'f1': f1}
#     if full_stats:
#         if total_b > 0:
#             exact_match_b = 100.0 * exact_match_b / total_b
#             exact_match_4 = 100.0 * exact_match_4 / total_4
#             f1_b = 100.0 * f1_b / total_b
#             f1_4 = 100.0 * f1_4 / total_4
#             ret.update({'exact_match_b': exact_match_b, 'f1_b': f1_b,
#                 'exact_match_4': exact_match_4, 'f1_4': f1_4,
#                 'total_b': total_b, 'total_4': total_4, 'total': total})

#         ret['qaid2perf'] = qaid2perf

#     return ret

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


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

