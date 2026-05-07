import argparse
from typing import Any, Iterable, Optional

import pandas as pd
from transformers import AutoTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def to_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def assistant_text_from_row(row: pd.Series) -> Optional[str]:
    # Prefer "target" messages, then fallback to "prompt".
    for column in ("target", "prompt"):
        if column not in row:
            continue
        for message in to_iterable(row[column]):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, str):
                    return content
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter rows by assistant response token length, then randomly "
            "sample exactly 128 rows and save as parquet."
        )
    )
    parser.add_argument(
        "--input-parquet",
        default="LIMOv2.parquet", # https://huggingface.co/datasets/GAIR/LIMO-v2/viewer/default/train?row=0
        help="Input parquet path.",
    )
    parser.add_argument(
        "--output-parquet",
        default="LIMOv2_8192_full.parquet",
        help="Output parquet path.",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Hugging Face tokenizer name or local path.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum assistant response token length.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=257,
        help="Number of rows to sample from eligible rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input_parquet)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    eligible_indices = []
    iterator = df.iterrows()
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(df), desc="Filtering rows", unit="row")

    for idx, row in iterator:
        text = assistant_text_from_row(row)
        if text is None:
            continue

        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count <= args.max_tokens:
            eligible_indices.append(idx)

    if len(eligible_indices) < args.sample_size:
        raise ValueError(
            f"Not enough eligible rows: found {len(eligible_indices)}, "
            f"need {args.sample_size}."
        )

    eligible_df = df.loc[eligible_indices]
    sampled_df = eligible_df.sample(n=args.sample_size, random_state=args.seed)
    sampled_df.to_parquet(args.output_parquet, index=False)

    print(f"Input parquet: {args.input_parquet}")
    print(f"Output parquet: {args.output_parquet}")
    print(f"Total rows: {len(df)}")
    print(f"Eligible rows (assistant tokens <= {args.max_tokens}): {len(eligible_df)}")
    print(f"Sampled rows written: {len(sampled_df)}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
