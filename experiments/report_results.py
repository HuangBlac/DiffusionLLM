"""Create a factual Markdown report from completed, matched pilot runs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    runs = {name: json.loads((ROOT / 'results' / f'qwen05b-{name}-200.json').read_text(encoding='utf-8'))
            for name in ('shift', 'same')}
    for run in runs.values():
        assert run['status'] == 'completed'
        assert len(run['training_log']) == run['steps'] == 200
        assert run['adapter_max_change'] > 0
    for key in ('seed', 'steps', 'batch_size', 'training_tokens', 'test_examples', 'data_manifest_sha256', 'script_sha256'):
        assert runs['shift'][key] == runs['same'][key], key
    for a, b in zip(runs['shift']['training_log'], runs['same']['training_log']):
        assert a['mask_probability'] == b['mask_probability'] and a['masked_tokens'] == b['masked_tokens']
    lines = ['# Qwen2.5 0.5B masked denoising pilot', '',
             'Completed on 2026-11-01. Single seed; 200 updates and 25,600 token exposures per condition. '
             'Both conditions start from the same real pretrained weights, use the same trainable parameter set, paragraph order and corruptions. '
             'The fixed final checkpoints are evaluated on 32 held-out 128-token sequences. '
             'Held out refers to adaptation, not to the unknown pretraining corpus.', '',
             '## Denoising results', '',
             '| Prediction | Mask probability | CE before | CE after | Accuracy before | Accuracy after | Masked test tokens |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, run in runs.items():
        for probability in ('0.25','0.5','0.75'):
            before, after = run['before'][probability], run['after'][probability]
            assert before['masked_tokens'] == after['masked_tokens'] == runs['shift']['before'][probability]['masked_tokens']
            lines.append(f"| {name} | {probability} | {before['masked_ce_nats']:.4f} | {after['masked_ce_nats']:.4f} | "
                         f"{before['masked_accuracy']:.2%} | {after['masked_accuracy']:.2%} | {after['masked_tokens']} |")
    lines += ['', 'CE is cross-entropy in nats per masked target, not perplexity or exact likelihood. '
              'The prediction alignments have different initial behaviors. These numbers do not demonstrate general language quality.', '',
              '## Runtime and controls', '', '| Prediction | Training seconds | Peak allocated GiB | Trainable parameters |',
              '| --- | ---: | ---: | ---: |']
    for name, run in runs.items():
        lines.append(f"| {name} | {run['training_seconds']:.1f} | {run['peak_allocated_gib']:.3f} | {run['trainable_parameters']:,} |")
    ar = runs['shift']['ar_reference_before']['next_token_ce_nats']
    lines += ['', f"Hardware: {runs['shift']['gpu']}. Original causal-model next-token CE: {ar:.4f} nats. "
              'That reference sees different context from the denoising task and cannot be directly ranked against masked CE.', '',
              'Timing covers the training loop, including logging and device synchronization; excludes loading and evaluation. '
              'Peak memory covers training. Equal update/token budgets are not a claim of equal wall-clock compute.', '',
              '## Fixed prompt samples after adaptation', '',
              'Each continuation uses 32 tokens and eight rounds of confidence-based greedy reveal. '
              'All three predetermined prompts are included for each condition; none was selected for fluency. '
              'Before-adaptation samples and per-round remaining-mask counts are retained in the raw JSON.', '']
    for name, run in runs.items():
        lines += [f'### {name}', '']
        for sample in run['samples_after']:
            lines += [f"Prompt: {sample['prompt']}", '', '```text', sample['completion'], '```', '']
    lines += ['## Scope and next decision', '',
              'This is an AI-assisted implementation and execution of a small adaptation pilot. It does not reproduce Dream/CART, '
              'validate 7B adaptation, establish superiority over AR, or demonstrate independent mastery by the repository owner. '
              'A training or held-out loss improvement can coexist with poor iterative generation. '
              'Before scaling, inspect the samples and explain the alignment/loss/sampling implementation, then repeat across seeds and larger held-out sets.', '',
              'See [protocol](experiment-design.md), [shift raw results](../../results/qwen05b-shift-200.json), '
              '[same-position raw results](../../results/qwen05b-same-200.json), and [data manifest](../../results/data_manifest.json).', '']
    (ROOT / 'docs' / 'history' / 'pilot-results.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Verified matched conditions and wrote docs/history/pilot-results.md')


if __name__ == '__main__':
    main()
