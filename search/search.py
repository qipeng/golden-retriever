from elasticsearch import Elasticsearch
import json
import re

from utils.constant import WIKIPEDIA_INDEX_NAME

es = Elasticsearch(timeout=300)

core_title_matcher = re.compile('([^()]+[^\s()])(?:\s*\(.+\))?')
core_title_filter = lambda x: core_title_matcher.match(x).group(1) if core_title_matcher.match(x) else x

def _extract_one(item, lazy=False):
    res = {k: item['_source'][k] for k in ['id', 'url', 'title', 'text', 'title_unescape']}
    res['_score'] = item['_score']
    res['data_object'] = item['_source']['original_json'] if lazy else json.loads(item['_source']['original_json'])

    return res

def _single_query_constructor(query, topn=50):
    return {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^1.25", "title_unescape^1.25", "text", "title_bigram^1.25", "title_unescape_bigram^1.25", "text_bigram"]
                }
            },
            "size": topn
        }

def single_text_query(query, topn=10, lazy=False, rerank_topn=50):
    body = _single_query_constructor(query, topn=max(topn, rerank_topn))
    res = es.search(index=WIKIPEDIA_INDEX_NAME, doc_type='doc', body=json.dumps(body))

    res = [_extract_one(x, lazy=lazy) for x in res['hits']['hits']]
    res = rerank_with_query(query, res)[:topn]

    return res

def bulk_text_query(queries, topn=10, lazy=False, rerank_topn=50):
    body = ["{}\n" + json.dumps(_single_query_constructor(query, topn=max(topn, rerank_topn))) for query in queries]
    res = es.msearch(index=WIKIPEDIA_INDEX_NAME, doc_type='doc', body='\n'.join(body))

    res = [[_extract_one(x, lazy=lazy) for x in r['hits']['hits']] for r in res['responses']]
    res = [rerank_with_query(query, results)[:topn] for query, results in zip(queries, res)]

    return res

def rerank_with_query(query, results):
    def score_boost(item, query):
        score = item['_score']
        core_title = core_title_filter(item['title_unescape'])
        if query.startswith('The ') or query.startswith('the '):
            query1 = query[4:]
        else:
            query1 = query
        if query == item['title_unescape'] or query1 == item['title_unescape']:
            score *= 1.5
        elif query.lower() == item['title_unescape'].lower() or query1.lower() == item['title_unescape'].lower():
            score *= 1.2
        elif item['title'].lower() in query:
            score *= 1.1
        elif query == core_title or query1 == core_title:
            score *= 1.2
        elif query.lower() == core_title.lower() or query1.lower() == core_title.lower():
            score *= 1.1
        elif core_title.lower() in query.lower():
            score *= 1.05

        item['_score'] = score
        return item

    return list(sorted([score_boost(item, query) for item in results], key=lambda item: -item['_score']))

if __name__ == "__main__":
    print([x['title'] for x in single_text_query("In which city did Mark Zuckerberg go to college?")])
    print([[y['title'] for y in x] for x in bulk_text_query(["In which city did Mark Zuckerberg go to college?"])])
