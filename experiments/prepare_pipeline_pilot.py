"""Prepare a development-only full-parameter pilot from already downloaded data."""
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from experiments.full_parameter_pilot import ROOT, sha256


def examples(texts, tokenizer, limit):
    rows, title = [], None
    for index, text in enumerate(texts):
        stripped = text.strip()
        if re.fullmatch(r"= [^=].*[^=] =", stripped):
            title = stripped
            continue
        if not stripped or stripped.startswith("=") or title is None:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 136:
            continue
        # EOS supervises an actual paragraph boundary, never an arbitrary crop.
        if len(ids) < 256:
            ids.append(tokenizer.eos_token_id)
        rows.append({"source_row": index, "article": title, "prefix_length": 128,
                     "ids": ids[:256], "natural_end": len(ids) <= 256 and
                     ids[-1] == tokenizer.eos_token_id})
        if len(rows) == limit:
            break
    if len(rows) != limit:
        raise ValueError(f"Need {limit} eligible paragraphs, got {len(rows)}")
    return rows


def main():
    destination = ROOT / "data" / "pipeline-pilot.json"
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "data" / "model", local_files_only=True)
    old = json.loads((ROOT / "results" / "data_manifest.json").read_text(encoding="utf-8"))
    data = {"status": "observed_development_pilot", "model_sha256": old["weights_sha256"],
            "model_revision": old["model_revision"], "dataset_revision": old["dataset_revision"],
            "subset": "wikitext-2-raw-v1", "sources": {}}
    for split, size in (("train", 512), ("validation", 32)):
        source = ROOT / "data" / "corpus" / data["subset"] / f"{split}-00000-of-00001.parquet"
        data["sources"][split] = sha256(source)
        data[split] = examples(pq.read_table(source, columns=["text"])["text"].to_pylist(), tokenizer, size)
    overlap = {r["article"] for r in data["train"]} & {r["article"] for r in data["validation"]}
    if overlap:
        raise ValueError("Pilot train/development articles overlap")
    encoded = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != encoded:
        raise FileExistsError("Different pilot data already exists; do not overwrite it")
    destination.write_text(encoded, encoding="utf-8")
    summary = {"status": data["status"], "data_sha256": sha256(destination),
               "model_sha256": data["model_sha256"], "sources": data["sources"],
               "limitations": "Previously observed WikiText-2; not the final held-out test",
               "splits": {split: {"paragraphs": len(data[split]),
                                    "articles": len({r['article'] for r in data[split]}),
                                    "natural_end_examples": sum(r['natural_end'] for r in data[split]),
                                    "source_rows": [r['source_row'] for r in data[split]]}
                          for split in ("train", "validation")}}
    (ROOT / "results" / "pipeline-pilot-data.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sha256": summary["data_sha256"],
                      "splits": {s: {k: v for k, v in d.items() if k != "source_rows"}
                                 for s, d in summary["splits"].items()}}))


if __name__ == "__main__":
    main()
