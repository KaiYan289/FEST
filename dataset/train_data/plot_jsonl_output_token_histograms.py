import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def _is_correct_rollout(row: dict) -> Optional[bool]:
    acc = row.get("acc")
    if isinstance(acc, bool):
        return acc

    score = row.get("score")
    if isinstance(score, (int, float)):
        if float(score) == 1.0:
            return True
        if float(score) == 0.0:
            return False

    return None


def collect_output_token_lengths(
    jsonl_path: Path, tokenizer: AutoTokenizer
) -> Tuple[List[int], List[int]]:
    correct_lengths: List[int] = []
    incorrect_lengths: List[int] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping invalid JSON in {jsonl_path.name}:{line_number}")
                continue

            output = row.get("output")
            if not isinstance(output, str):
                continue

            is_correct = _is_correct_rollout(row)
            if is_correct is None:
                continue

            token_count = len(tokenizer.encode(output, add_special_tokens=False))
            if is_correct:
                correct_lengths.append(token_count)
            else:
                incorrect_lengths.append(token_count)
    return correct_lengths, incorrect_lengths


def save_histogram(
    correct_lengths: List[int],
    incorrect_lengths: List[int],
    jsonl_path: Path,
    output_dir: Path,
    bins: int,
) -> None:
    if not correct_lengths and not incorrect_lengths:
        print(f"[WARN] No valid output texts with acc/score in {jsonl_path.name}; skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    axes[0].hist(correct_lengths, bins=bins, color="#2ca02c", edgecolor="black")
    axes[0].set_title(f"Correct (n={len(correct_lengths)})")
    axes[0].set_xlabel("Token length")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].hist(incorrect_lengths, bins=bins, color="#d62728", edgecolor="black")
    axes[1].set_title(f"Incorrect (n={len(incorrect_lengths)})")
    axes[1].set_xlabel("Token length")
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle(
        (
            "Output Token Length Histogram by Rollout Correctness: "
            f"{jsonl_path.name}\nTokenizer: Qwen/Qwen2.5-Math-1.5B"
        )
    )
    plt.tight_layout()

    output_path = output_dir / f"{jsonl_path.stem}_output_token_hist_by_correctness.png"
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "For each JSONL in a directory, draw histogram of token lengths "
            "for 'output' field using Qwen2.5-Math-1.5B tokenizer."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=(
            "../../rollout_data/"
            "fromdeltaai-v3_smartsftrl_gw0.05_wl0.01_gl0.005-bugfixed-256shot"
        ),
        help="Directory containing JSONL files.",
    )
    parser.add_argument(
        "--output-subdir",
        default="output_token_histograms",
        help="Subfolder name under input-dir to store histogram images.",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Hugging Face tokenizer name or local path.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of histogram bins.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found under: {input_dir}")

    output_dir = input_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    iterator = jsonl_files
    if tqdm is not None:
        iterator = tqdm(jsonl_files, desc="Processing JSONL files", unit="file")

    processed = 0
    skipped = 0
    for jsonl_path in iterator:
        correct_lengths, incorrect_lengths = collect_output_token_lengths(jsonl_path, tokenizer)
        if not correct_lengths and not incorrect_lengths:
            skipped += 1
            continue
        save_histogram(correct_lengths, incorrect_lengths, jsonl_path, output_dir, args.bins)
        processed += 1

    print(f"Input directory: {input_dir}")
    print(f"Output histogram directory: {output_dir}")
    print(f"JSONL files discovered: {len(jsonl_files)}")
    print(f"Histograms saved: {processed}")
    print(f"Files skipped (no valid output values): {skipped}")


if __name__ == "__main__":
    main()
