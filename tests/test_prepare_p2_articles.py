import hashlib

import pytest
from experiments.prepare_p2_articles import articles, canonical_title, heading_title, partition


def test_heading_and_shard_continuation_preserve_article_boundaries():
    rows = [('a', 0, '\n'), ('a', 1, ' = Alpha = \n'), ('a', 2, '\n'), ('a', 3, 'first\n'),
            ('a', 4, ' = = Section = = \n'), ('b', 0, 'continued\n'),
            ('b', 1, '\n'), ('b', 2, ' = Beta = \n'), ('b', 3, '\n'), ('b', 4, 'second\n')]
    result = list(articles(rows))
    assert [r['title'] for r in result] == ['Alpha', 'Beta']
    assert result[0]['source_spans'] == [{'source': 'a', 'start': 1, 'end': 4},
                                         {'source': 'b', 'start': 0, 'end': 1}]
    assert result[0]['nonblank_rows'] == 4
    changed = rows.copy()
    changed[5] = ('b', 0, 'different\n')
    assert list(articles(changed))[0]['raw_text_sha256'] != result[0]['raw_text_sha256']


def record(title, text):
    return {'title': title, 'canonical_title': canonical_title(title),
            'raw_text_sha256': hashlib.sha256((title+'\0'+text).encode()).hexdigest(),
            'content_sha256': hashlib.sha256(text.encode()).hexdigest(), 'source_spans': []}


def test_exclusion_deduplication_and_split_assignment_are_order_independent():
    assert canonical_title(' = Title0 = ') == canonical_title('title0')
    records = [record(f'Title{i}', str(i)) for i in range(12)]
    records += [record('TITLE1', 'another'), record('copy', '2')]
    result, removed = partition(records, [' Title0 '], development=2, final=3)
    reverse, reverse_removed = partition(list(reversed(records)), ['title0'], development=2, final=3)
    assert result == reverse and removed == reverse_removed
    assert removed == {'wt2_overlap': 1, 'wt2_content_overlap': 0, 'empty_article': 0,
                       'duplicate_title': 1, 'duplicate_text': 1}
    assert len({r['canonical_title'] for r in result}) == len(result)
    assert len({r['content_sha256'] for r in result}) == len(result)
    assert sum(r['split'] == 'development' for r in result) == 2
    assert sum(r['split'] == 'final_continuation' for r in result) == 3
    assert all(r['canonical_title'] != 'title0' for r in result)


def test_orphan_content_or_insufficient_articles_fails_before_freezing():
    with pytest.raises(ValueError, match='before the first'):
        list(articles([('a', 0, 'orphan')]))
    with pytest.raises(ValueError, match='Not enough'):
        partition([record(str(i), str(i)) for i in range(5)], [], development=2, final=3)


def test_body_duplicates_ignore_title_and_exact_duplicates_choose_stable_provenance():
    rows = [('a', 0, ' = Alpha = \n'), ('a', 1, '\n'), ('a', 2, 'same body\n'),
            ('a', 3, '\n'), ('a', 4, ' = Beta = \n'), ('a', 5, '\n'), ('a', 6, 'same body\n')]
    parsed = list(articles(rows))
    assert parsed[0]['raw_text_sha256'] != parsed[1]['raw_text_sha256']
    assert parsed[0]['content_sha256'] == parsed[1]['content_sha256']
    records = parsed+[record(f'Other{i}', str(i)) for i in range(5)]
    duplicate = {**records[-1], 'source_spans': [{'source': 'z', 'start': 0, 'end': 0}]}
    records.append(duplicate)
    a, removed = partition(records, [], development=1, final=1)
    b, _ = partition(list(reversed(records)), [], development=1, final=1)
    assert a == b and removed['duplicate_text'] == 1 and removed['duplicate_title'] == 1


def test_internal_equals_is_a_title_but_section_markers_are_not():
    assert heading_title(' = E = mc² = \n') == 'E = mc²'
    assert heading_title(' = = Section = = \n') is None
    assert heading_title(' = = = Subsection = = = \n') is None


def test_wt2_body_under_a_different_title_and_empty_articles_are_excluded():
    records = [record(str(i), str(i)) for i in range(6)]
    records += [record('new title', 'old observed content'), {**record('empty', ''), 'body_nonblank_rows': 0}]
    result, removed = partition(records, [], excluded_content_hashes=[hashlib.sha256(b'old observed content').hexdigest()],
                                development=1, final=1)
    assert len(result) == 6
    assert removed['wt2_content_overlap'] == 1 and removed['empty_article'] == 1


def test_table_legend_is_body_while_isolated_title_can_span_shards():
    rows = [('a', 0, ' = Article = \n'), ('a', 1, ''), ('a', 2, 'rules for classification\n'),
            ('a', 3, ' = Champion ; ( R ) = \n'), ('a', 4, 'Relegated\n'),
            ('a', 5, ''), ('a', 6, ' = Next article = \n'), ('b', 0, ''), ('b', 1, 'next body\n')]
    parsed = list(articles(rows))
    assert [r['title'] for r in parsed] == ['Article', 'Next article']
    assert parsed[0]['body_nonblank_rows'] == 3
