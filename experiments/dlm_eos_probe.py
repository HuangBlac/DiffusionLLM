"""P2 exploratory EOS/canvas check: same CPU weights, greedy C8 at three lengths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from experiments.dlm_generation_diagnosis import decode, save_report, select_rows
from experiments.full_parameter_pilot import ConditionalModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    if args.output.exists() or time.monotonic() >= args.deadline:
        raise ValueError('Use a fresh output and future deadline')
    root = Path(__file__).resolve().parents[1]
    identity = json.loads(args.manifest.read_text(encoding='utf-8'))
    if hashlib.sha256((root/'experiments/full_parameter_pilot.py').read_bytes()).hexdigest() != identity['identity']['script_sha256']:
        raise ValueError('Frozen inference kernel changed')
    with args.weights.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != identity['weights']['sha256']:
            raise ValueError('Checkpoint checksum mismatch')
    data_bytes = (root/'data/pipeline-pilot.json').read_bytes()
    if hashlib.sha256(data_bytes).hexdigest() != identity['identity']['data_sha256']:
        raise ValueError('Frozen diagnostic data changed')
    rows = select_rows(json.loads(data_bytes)['validation'], 8)
    report = {'status': 'running', 'training_performed': False, 'device': 'cpu', 'precision': 'FP32',
              'torch': str(torch.__version__), 'transformers': transformers.__version__,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_checkpoint_sha256': identity['weights']['sha256'],
              'purpose': 'exploratory_P2_probe_triggered_by_EOS_at_canvas_end',
              'lengths': [32, 64, 128], 'source_rows': [r['source_row'] for r in rows], 'samples': []}
    save_report(args.output, report)
    try:
        torch.set_num_threads(4)
        config = AutoConfig.from_pretrained(root/'data/model', local_files_only=True)
        model = ConditionalModel(AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32,
                                  attn_implementation='sdpa'), 'dlm').requires_grad_(False).eval()
        state = torch.load(args.weights, map_location='cpu', weights_only=True, mmap=True)
        if state['step'] != 200 or state['identity'] != identity['identity']:
            raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(state['model'], strict=True)
        del state
        tokenizer = AutoTokenizer.from_pretrained(root/'data/model', local_files_only=True)
        for row in rows:
            for length in report['lengths']:
                result = decode(model, tokenizer, row, order='confidence', group=8,
                                temperature=0, seed=101, length=length, deadline=args.deadline)
                report['samples'].append(result)
                save_report(args.output, report)
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='partial', error_type=type(error).__name__)
        raise
    finally:
        save_report(args.output, report)


if __name__ == '__main__':
    main()
