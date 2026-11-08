"""Build exact-deduplicated development inputs from article candidates, without training."""
import argparse
import hashlib
import json
from pathlib import Path
import unicodedata


def paragraph_hash(text):
    normalized = unicodedata.normalize('NFKC', ' '.join(text.split()))
    return hashlib.sha256(normalized.encode()).hexdigest()


def window(ids, eos):
    if len(ids) < 136:
        return None
    natural = len(ids) < 256
    return {'ids': (ids+[eos] if natural else ids)[:256], 'prefix_length': 128,
            'natural_end': natural}


def select_examples(articles, rows, encode, eos, excluded_paragraphs, *, training=4096,
                    development=100, final=300, calibration=60, excluded_prefixes=()):
    """Choose first eligible non-WT2 paragraph per article, then dedup before splitting."""
    counts = {'training': training, 'development': development,
              'final_continuation': final, 'calibration_sources': calibration}
    if any(type(n) is not int or n <= 0 for n in counts.values()):
        raise ValueError('Need positive fixed split counts')
    article_map = {a['article_id']: a for a in articles}
    if len(article_map) != len(articles):
        raise ValueError('Duplicate article IDs')
    ranges = {}
    for article in articles:
        for span in article['source_spans']:
            ranges.setdefault(span['source'], []).append((span['start'], span['end'], article['article_id']))
    for source, spans in ranges.items():
        spans.sort()
        if any(start <= previous_end for (_, previous_end, _), (start, _, _) in zip(spans, spans[1:])):
            raise ValueError('Overlapping article owners in '+source)
    pointers = {source: 0 for source in ranges}
    chosen = {}
    excluded_prefixes = set(excluded_prefixes)
    observed_prefix_rows = 0
    last_indices = {}
    for source, index, text in rows:
        if source in last_indices and index <= last_indices[source]:
            raise ValueError('Source row order is not strictly increasing')
        last_indices[source] = index
        spans = ranges.get(source, [])
        pointer = pointers.get(source, 0)
        while pointer < len(spans) and spans[pointer][1] < index:
            pointer += 1
        pointers[source] = pointer
        if pointer == len(spans) or spans[pointer][0] > index:
            continue
        owner = spans[pointer][2]
        if owner in chosen or not text.strip() or text.lstrip().startswith('='):
            continue
        digest = paragraph_hash(text)
        if digest in excluded_paragraphs:
            continue
        value = window(encode(text), eos)
        if value is None:
            continue
        prefix_hash = hashlib.sha256(json.dumps(value['ids'][:128]).encode()).hexdigest()
        if prefix_hash in excluded_prefixes:
            observed_prefix_rows += 1
            continue
        chosen[owner] = {**value, 'article_id': owner, 'article': article_map[owner]['title'],
                         'source': source, 'source_row': index, 'paragraph_sha256': digest,
                         'prefix_sha256': prefix_hash}
    unique = []
    seen_paragraphs, seen_prefixes = set(), set()
    removed = {'no_eligible_paragraph': len(articles)-len(chosen), 'duplicate_paragraph': 0, 'duplicate_prefix': 0,
               'observed_prefix_rows_skipped_before_choice': observed_prefix_rows}
    for owner in sorted(chosen):
        example = chosen[owner]
        if example['paragraph_sha256'] in seen_paragraphs:
            removed['duplicate_paragraph'] += 1
        elif example['prefix_sha256'] in seen_prefixes:
            removed['duplicate_prefix'] += 1
        else:
            unique.append(example)
            seen_paragraphs.add(example['paragraph_sha256'])
            seen_prefixes.add(example['prefix_sha256'])
    if len(unique) < sum(counts.values()):
        raise ValueError('Insufficient eligible articles after paragraph and prefix deduplication')
    result = {}
    start = 0
    for split in ('final_continuation', 'development', 'calibration_sources', 'training'):
        result[split] = unique[start:start+counts[split]]
        start += counts[split]
    return result, removed


def parquet_rows(data_root, source_names):
    import pyarrow.parquet as pq
    for source in sorted(source_names):
        path = (data_root/source).resolve()
        if not path.is_relative_to(data_root.resolve()):
            raise ValueError('Unsafe source filename')
        index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=['text'], use_threads=False):
            for text in batch.column(0).to_pylist():
                yield source, index, text
                index += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--articles', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    manifest = json.loads(args.articles.read_text(encoding='utf-8'))
    if manifest['status'] != 'article_split_candidates_not_frozen':
        raise ValueError('Expected article candidates')
    for source, expected in manifest['source_hashes'].items():
        path = (args.data_root/source).resolve()
        if not path.is_relative_to(args.data_root.resolve()):
            raise ValueError('Unsafe source filename')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError('Pinned source file changed')
    from tokenizers import Tokenizer
    import tokenizers
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids
    eos = tokenizer.token_to_id('<|endoftext|>')
    if eos is None:
        raise ValueError('Missing EOS token')
    pilot = json.loads(args.pilot.read_text(encoding='utf-8'))
    expected_rows = {('wikitext-2-raw-v1/'+split+'-00000-of-00001.parquet', row['source_row']): row
                     for split in ('train', 'validation') for row in pilot[split]}
    matched = 0
    excluded = set()
    excluded_prefixes = set()
    wt2_sources = [source for source in manifest['source_hashes'] if source.startswith('wikitext-2-raw-v1/')]
    for source, index, text in parquet_rows(args.data_root, wt2_sources):
        ids = None
        if text.strip() and not text.lstrip().startswith('='):
            excluded.add(paragraph_hash(text))
            ids = encode(text)
            if len(ids) >= 128:
                excluded_prefixes.add(hashlib.sha256(json.dumps(ids[:128]).encode()).hexdigest())
        if (source, index) in expected_rows:
            value = window(ids if ids is not None else encode(text), eos)
            if value is None or value['ids'] != expected_rows[(source, index)]['ids']:
                raise ValueError('Tokenizer differs from the frozen pilot')
            matched += 1
    if matched != len(expected_rows):
        raise ValueError('Missing tokenizer verification rows')
    wt103_sources = [source for source in manifest['source_hashes'] if source.startswith('wikitext-103-raw-v1/')]
    data, removed = select_examples(manifest['articles'], parquet_rows(args.data_root, wt103_sources),
                                      encode, eos, excluded, excluded_prefixes=excluded_prefixes)
    args.output.mkdir(parents=True)
    artifacts = {}
    for split, examples in data.items():
        payload = (json.dumps(examples, ensure_ascii=False, indent=2)+'\n').encode()
        (args.output/(split+'.json')).write_bytes(payload)
        artifacts[split] = {'examples': len(examples), 'sha256': hashlib.sha256(payload).hexdigest(),
                            'bytes': len(payload), 'natural_end_examples': sum(e['natural_end'] for e in examples)}
    summary = {'status': 'development_inputs_prepared_not_formal_test_release', 'training_performed': False,
               'counts': {split: len(examples) for split, examples in data.items()}, 'artifacts': artifacts,
               'removed': removed, 'tokenizer_sha256': hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
               'tokenizers_version': tokenizers.__version__, 'pilot_rows_tokenization_exact': matched,
               'excluded_wt2_paragraph_hashes': len(excluded), 'excluded_wt2_prefix_hashes': len(excluded_prefixes),
               'article_candidate_sha256': hashlib.sha256(args.articles.read_bytes()).hexdigest(),
               'prefix_length': 128, 'window_cap': 256, 'minimum_source_tokens': 136,
               'limits': ['Near-duplicate/shared-fragment audit still pending', 'Calibration sources are not human labels',
                          'Reserved final texts must not be inspected or used for development selection',
                          'Base-model pretraining overlap is unknown']}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
