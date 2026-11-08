"""Prepare deterministic article candidates; token eligibility precedes final freeze."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

HEADING = re.compile(r'=\s+(.+?)\s+=')


def heading_title(text):
    match = HEADING.fullmatch(text.strip())
    if not match:
        return None
    title = match.group(1).strip()
    return None if title.startswith('=') and title.endswith('=') else title


def canonical_title(title):
    title = unicodedata.normalize('NFKC', title).strip()
    heading = heading_title(title)
    if heading is not None:
        title = heading
    return ' '.join(title.split()).casefold()


def articles(rows):
    """WikiText article titles are isolated by blank rows, including across shards."""
    current = None
    digest = None
    content_digest = None
    iterator = iter(rows)
    before_blank = True
    row = next(iterator, None)
    while row is not None:
        after = next(iterator, None)
        source, index, text = row
        title = heading_title(text) if before_blank and after is not None and not after[2].strip() else None
        before_blank = not text.strip()
        row = after
        if title is not None:
            if current is not None:
                current['raw_text_sha256'] = digest.hexdigest()
                current['content_sha256'] = content_digest.hexdigest()
                yield current
            current = {'title': title, 'canonical_title': canonical_title(title),
                       'characters': 0, 'nonblank_rows': 0, 'body_nonblank_rows': 0, 'source_spans': []}
            digest = hashlib.sha256()
            content_digest = hashlib.sha256()
        if current is None:
            if text.strip():
                raise ValueError('Nonblank content before the first top-level heading')
            continue
        encoded = text.encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'big'))
        digest.update(encoded)
        if title is None and text.strip():
            normalized = ' '.join(text.split()).encode('utf-8')
            content_digest.update(len(normalized).to_bytes(8, 'big'))
            content_digest.update(normalized)
            current['body_nonblank_rows'] += 1
        current['characters'] += len(text)
        current['nonblank_rows'] += bool(text.strip())
        spans = current['source_spans']
        if spans and spans[-1]['source'] == source and spans[-1]['end']+1 == index:
            spans[-1]['end'] = index
        else:
            spans.append({'source': source, 'start': index, 'end': index})
    if current is not None:
        current['raw_text_sha256'] = digest.hexdigest()
        current['content_sha256'] = content_digest.hexdigest()
        yield current


def partition(records, excluded_titles, *, excluded_content_hashes=(), development=100, final=300, salt='p2-v1'):
    if development < 1 or final < 1:
        raise ValueError('Need positive development and final article counts')
    excluded = {canonical_title(title) for title in excluded_titles}
    excluded_content_hashes = set(excluded_content_hashes)
    # Dedup before assigning sets; duplicate headings/content never cross splits.
    unique = []
    seen_titles, seen_text = set(), set()
    removed = {'wt2_overlap': 0, 'wt2_content_overlap': 0, 'empty_article': 0,
               'duplicate_title': 0, 'duplicate_text': 0}
    for record in sorted(records, key=lambda r: (r['canonical_title'], r['raw_text_sha256'],
                                                 json.dumps(r['source_spans'], sort_keys=True))):
        title, text_hash = record['canonical_title'], record['content_sha256']
        if title in excluded:
            removed['wt2_overlap'] += 1
        elif text_hash in excluded_content_hashes:
            removed['wt2_content_overlap'] += 1
        elif record.get('body_nonblank_rows', 1) == 0:
            removed['empty_article'] += 1
        elif title in seen_titles:
            removed['duplicate_title'] += 1
        elif text_hash in seen_text:
            removed['duplicate_text'] += 1
        else:
            seen_titles.add(title)
            seen_text.add(text_hash)
            unique.append(record)
    if len(unique) <= development+final:
        raise ValueError('Not enough distinct articles after exclusions')
    def key(record):
        return hashlib.sha256((salt+'\0'+record['canonical_title']+'\0'+record['raw_text_sha256']).encode()).hexdigest()
    unique.sort(key=key)
    result = []
    for index, record in enumerate(unique):
        split = 'final_continuation' if index < final else (
            'development' if index < final+development else 'training')
        result.append({**record, 'article_id': key(record), 'split': split})
    return result, removed


def parquet_articles(paths, label):
    import pyarrow.parquet as pq

    def rows():
        for path in paths:
            index = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=['text'], use_threads=False):
                for text in batch.column(0).to_pylist():
                    yield label+'/'+path.name, index, text
                    index += 1
    return list(articles(rows()))


def corpus(directory, label):
    records, source_hashes = [], {}
    for split in ('train', 'validation', 'test'):
        paths = sorted(directory.glob(split+'-*.parquet'))
        if not paths:
            raise FileNotFoundError('Missing source split '+split)
        for path in paths:
            with path.open('rb') as stream:
                source_hashes[label+'/'+path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        records.extend(parquet_articles(paths, label))
    return records, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wt103', type=Path, required=True)
    parser.add_argument('--wt2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    original, original_hashes = corpus(args.wt2, 'wikitext-2-raw-v1')
    records, hashes = corpus(args.wt103, 'wikitext-103-raw-v1')
    assigned, removed = partition(records, [r['title'] for r in original],
                                  excluded_content_hashes=[r['content_sha256'] for r in original])
    manifest = {'status': 'article_split_candidates_not_frozen', 'schema': 1,
                'eligibility_status': 'pending_tokenization_and_continuation_length_check',
                'training_performed': False, 'source_hashes': {**original_hashes, **hashes},
                'preparer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'heading_pattern': HEADING.pattern,
                'heading_section_rule': 'Blank row before and after; reject paired section equals markers',
                'body_hash_rule': 'Per nonblank row normalized whitespace; root heading excluded',
                'split_salt': 'p2-v1', 'excluded_wt2_titles': len({r['canonical_title'] for r in original}),
                'removed': removed, 'articles': assigned,
                'limits': ['Self-defined article holdouts, not official test splits',
                           'Base-model pretraining contamination unknown',
                           'File hashes identify these inputs; upstream download revision must also be recorded',
                           'Shared-paragraph and near-duplicate checks are still pending',
                           'Judge calibration and tokenization still need a frozen configuration']}
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode()
    args.output.mkdir(parents=True)
    (args.output/'article-manifest.json').write_bytes(payload)
    summary = {'status': manifest['status'], 'manifest_sha256': hashlib.sha256(payload).hexdigest(),
               'article_counts': {s: sum(r['split'] == s for r in assigned)
                                  for s in ('training', 'development', 'final_continuation')},
               'removed': removed, 'training_performed': False}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
