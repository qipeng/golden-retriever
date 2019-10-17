
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def make_context(question, ir_results):
    """
    Creates a single context string.

    :question: string
    :ir_results: list of dictionaries objects each of which
        should have 'title' and 'text'
        (e.g. each entry of result from bulk_text_query)
    """
    return question + ' ' + concat_paragraphs(ir_results)

def concat_paragraphs(ir_results):
    return ' '.join([f"<t> {p['title']} </t> {''.join(p['text'])}" for p in ir_results])
