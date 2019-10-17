from argparse import ArgumentParser
import bz2
from collections import Counter, defaultdict
from elasticsearch import Elasticsearch
from glob import glob
import html
import json
from multiprocessing import Pool
import numpy as np
import os
import pickle
import re
from tqdm import tqdm
from urllib.parse import unquote

from utils.constant import WIKIPEDIA_INDEX_NAME
from utils.general import chunks

def process_line(line):
    data = json.loads(line)
    item = {'id': data['id'],
            'url': data['url'],
            'title': data['title'],
            'title_unescape': html.unescape(data['title']),
            'text': ''.join(data['text']),
            'title_bigram': html.unescape(data['title']),
            'title_unescape_bigram': html.unescape(data['title']),
            'text_bigram': ''.join(data['text']),
            'original_json': line
            }
    # tell elasticsearch we're indexing documents
    return "{}\n{}".format(json.dumps({ 'index': { '_id': 'wiki-{}'.format(data['id']) } }), json.dumps(item))

def generate_indexing_queries_from_bz2(bz2file, dry=False):
    if dry:
        return

    with bz2.open(bz2file, 'rt') as f:
        body = [process_line(line) for line in f]

    return '\n'.join(body)

es = Elasticsearch(timeout=100)
def index_chunk(chunk):
    res = es.bulk(index=WIKIPEDIA_INDEX_NAME, doc_type='doc', body='\n'.join(chunk), timeout='100s')
    assert not res['errors'], res

def main(args):
    # make index
    if not args.dry:
        if es.indices.exists(index=WIKIPEDIA_INDEX_NAME) and args.reindex:
            es.indices.delete(index=WIKIPEDIA_INDEX_NAME, ignore=[400,403])
        if not es.indices.exists(index=WIKIPEDIA_INDEX_NAME):
            es.indices.create(index=WIKIPEDIA_INDEX_NAME, ignore=400,
                    body=json.dumps({
                        "mappings":{"doc":{"properties": {
                            "id": { "type": "keyword" },
                            "url": { "type": "keyword" },
                            "title": { "type": "text", "analyzer": "simple", "copy_to": "title_all"},
                            "title_unescape": { "type": "text", "analyzer": "simple", "copy_to": "title_all"},
                            "text": { "type": "text", "analyzer": "my_english_analyzer"},
                            "anchortext": { "type": "text", "analyzer": "my_english_analyzer"},
                            "title_bigram": { "type": "text", "analyzer": "simple_bigram_analyzer", "copy_to": "title_all_bigram"},
                            "title_unescape_bigram": { "type": "text", "analyzer": "simple_bigram_analyzer", "copy_to": "title_all_bigram"},
                            "text_bigram": { "type": "text", "analyzer": "bigram_analyzer"},
                            "anchortext_bigram": { "type": "text", "analyzer": "bigram_analyzer"},
                            "original_json": { "type": "string" },
                            }}
                        },
			"settings": {
                            "analysis": {
                                "my_english_analyzer": {
                                    "type": "standard",
                                    "stopwords": "_english_",
                                },
                                "simple_bigram_analyzer": {
                                    "tokenizer": "standard",
                                    "filter": [
                                         "lowercase", "shingle", "asciifolding"
                                    ]
                                },
                                "bigram_analyzer": {
                                    "tokenizer": "standard",
                                    "filter": [
                                         "lowercase", "stop", "shingle", "asciifolding"
                                    ]
                                }
                            },
                        }
                        }))

    filelist = glob('data/enwiki-20171001-pages-meta-current-withlinks-abstracts/*/wiki_*.bz2')

    print('Making indexing queries...')
    pool = Pool()
    all_queries = list(tqdm(pool.imap(generate_indexing_queries_from_bz2, filelist), total=len(filelist)))

    count = sum(len(queries.split('\n')) for queries in all_queries) // 2

    if not args.dry:
        print('Indexing...')
        chunksize = 50
        for chunk in tqdm(chunks(all_queries, chunksize), total=(len(all_queries) + chunksize - 1) // chunksize):
            res = es.bulk(index=WIKIPEDIA_INDEX_NAME, doc_type='doc', body='\n'.join(chunk), timeout='100s')
            assert not res['errors'], res

    print(f"{count} documents indexed in total")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--reindex', action='store_true', help="Reindex everything")
    parser.add_argument('--dry', action='store_true', help="Dry run")

    args = parser.parse_args()

    main(args)
