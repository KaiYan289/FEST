#from datasets import load_dataset
import pandas as pd

dataset = load_dataset("Elliott/Openr1-Math-46k-8192", split="train")

# print(dataset[0])

items = []

for i in range(len(dataset)):
    items.append(dataset['prompt'][i][1:])

dataset['prompt'] = items
dataset.to_parquet("openr1_hpt.parquet")
