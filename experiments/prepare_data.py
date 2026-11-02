"""Download versioned public artifacts; keep weights and corpus out of Git."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download, snapshot_download
import pyarrow.parquet as pq
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
DATASET = "Salesforce/wikitext"
DATA_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WEIGHTS_SHA256 = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
ROOT = Path(__file__).resolve().parents[1]


def main():
    (ROOT / 'results').mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "data" / "model"
    weights = model_dir / "model.safetensors"
    patterns = ["*.json", "*.txt", "*.model", "LICENSE*"]
    if not weights.exists():
        patterns.append("*.safetensors")
    snapshot_download(MODEL, revision=MODEL_REVISION, local_dir=model_dir,
                      allow_patterns=patterns,
                      max_workers=2)
    with weights.open("rb") as file:
        weights_hash = hashlib.file_digest(file, "sha256").hexdigest()
    if weights_hash != WEIGHTS_SHA256:
        raise RuntimeError("Pretrained weights SHA-256 does not match the pinned version")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    metadata = {"model": MODEL, "model_revision": MODEL_REVISION,
                "dataset": DATASET, "dataset_revision": DATA_REVISION,
                "subset": "wikitext-2-raw-v1", "length": 128,
                "weights_sha256": weights_hash, "splits": {}}
    # Full-paragraph hashes exclude exact overlap between adaptation splits.
    seen = set()
    seen_tokens = set()
    for split, count in [("train", 1024), ("validation", 64), ("test", 64)]:
        file = hf_hub_download(DATASET, f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
                               repo_type="dataset", revision=DATA_REVISION,
                               local_dir=ROOT / "data" / "corpus")
        texts = pq.read_table(file, columns=["text"]).column("text").to_pylist()
        examples, rows = [], []
        for row, text in enumerate(texts):
            digest = hashlib.sha256(text.strip().encode()).hexdigest()
            if digest in seen:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 128:
                continue
            token_key = tuple(ids[:128])
            if token_key in seen_tokens:
                continue
            seen.add(digest)
            seen_tokens.add(token_key)
            examples.append(ids[:128])
            rows.append(row)
            if len(examples) == count:
                break
        if len(examples) != count:
            raise RuntimeError(f"Only {len(examples)} eligible paragraphs in {split}")
        out = ROOT / "data" / f"{split}.json"
        out.write_text(json.dumps(examples), encoding="utf-8")
        metadata["splits"][split] = {"examples": count, "source_rows": rows,
                                      "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
        print(f"Prepared {split}: {count} x 128 tokens", flush=True)
    (ROOT / "results" / "data_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
