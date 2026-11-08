import hashlib
import json
import pytest
from experiments.prepare_p2_examples import paragraph_hash, select_examples, window


def fixture(n=10):
    articles = [{'article_id': f'{i:03}', 'title': str(i),
                 'source_spans': [{'source': 'a', 'start': i*3, 'end': i*3+2}]} for i in range(n)]
    rows = [(source, index, text) for i in range(n)
            for source, index, text in [('a', i*3, '= title ='), ('a', i*3+1, 'short'), ('a', i*3+2, f'{i}|good')]]
    return articles, rows


def encode(text):
    return [99]*135 if text == 'short' else [int(text.split('|')[0])+1]*200


def test_window_eos_only_at_true_paragraph_end():
    assert window([2]*135, 0) is None
    short = window([2]*255, 0)
    assert short['natural_end'] and len(short['ids']) == 256 and short['ids'][-1] == 0
    long = window([2]*300, 0)
    assert not long['natural_end'] and len(long['ids']) == 256 and long['ids'][-1] == 2


def test_eligible_rows_and_splits_exclude_observed_paragraphs_and_are_deterministic():
    articles, rows = fixture()
    counts = dict(training=2, development=1, final=1, calibration=1)
    data, removed = select_examples(articles, rows, encode, 0, {paragraph_hash('0|good')}, **counts)
    reverse, _ = select_examples(list(reversed(articles)), rows, encode, 0, {paragraph_hash('0|good')}, **counts)
    assert data == reverse and removed['no_eligible_paragraph'] == 1
    selected = [row for split in data.values() for row in split]
    assert len(selected) == len({r['article_id'] for r in selected}) == 5
    assert [r['article_id'] for r in selected] == ['001', '002', '003', '004', '005']
    assert all(r['source_row'] % 3 == 2 and len(r['ids']) == 201 for r in selected)


def test_identical_prefixes_and_paragraphs_are_removed_before_partition():
    articles, rows = fixture()
    rows[3*3+2] = ('a', 3*3+2, '2|good')
    def reused_prefix(text):
        values = encode(text)
        return [7]*128+values[128:] if text in ('6|good', '7|good') else values
    data, removed = select_examples(articles, rows, reused_prefix, 0, set(),
                                     training=2, development=1, final=1, calibration=1)
    assert removed['duplicate_paragraph'] == 1 and removed['duplicate_prefix'] == 1
    selected = [r for examples in data.values() for r in examples]
    assert len({r['paragraph_sha256'] for r in selected}) == len(selected)
    assert len({r['prefix_sha256'] for r in selected}) == len(selected)


def test_invalid_owners_row_order_and_short_corpus_fail_closed():
    articles, rows = fixture(2)
    with pytest.raises(ValueError, match='Insufficient'):
        select_examples(articles, rows, encode, 0, set(), training=1, development=1, final=1, calibration=1)
    with pytest.raises(ValueError, match='row order'):
        select_examples(articles, list(reversed(rows)), encode, 0, set(), training=1, development=1, final=1, calibration=1)
    articles.append({**articles[0], 'article_id': 'overlap'})
    with pytest.raises(ValueError, match='Overlapping'):
        select_examples(articles, rows, encode, 0, set(), training=1, development=1, final=1, calibration=1)


def test_observed_prefix_with_a_changed_tail_is_skipped_before_first_eligible_choice():
    articles, rows = fixture()
    rows[1] = ('a', 1, '0|observed')
    rows[2] = ('a', 2, '8|alternative')
    observed = hashlib.sha256(json.dumps([1]*128).encode()).hexdigest()
    data, removed = select_examples(articles, rows, encode, 0, set(), excluded_prefixes={observed},
                                     training=2, development=1, final=1, calibration=1)
    first = data['final_continuation'][0]
    assert first['article_id'] == '000' and first['source_row'] == 2 and first['ids'][0] == 9
    assert removed['observed_prefix_rows_skipped_before_choice'] == 1
