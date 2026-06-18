from datasets import load_dataset

dataset = load_dataset("Elliott/Openr1-Math-46k-8192", split="train")

dataset = dataset.map(lambda example: {"prompt": example["prompt"][1:]})

dataset.to_parquet("openr1_hpt.parquet")
