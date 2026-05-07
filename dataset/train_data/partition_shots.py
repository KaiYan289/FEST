import pandas as pd
import random

# We run this inside HPT codebase: https://github.com/TsinghuaC3I/Unify-Post-Training/tree/main/data
random.seed(42)

class RandomPartitioner:
    def __init__(self, dataset, partition_size):
        self.dataset = dataset
        self.partition_size = partition_size

    def partition(self):
        idx = random.sample(range(len(self.dataset)), self.partition_size)
        new_data = self.dataset.iloc[idx].reset_index(drop=True)
        return idx, pd.concat([new_data], ignore_index=True)

if __name__ == "__main__":
    dataset = pd.read_parquet("openr1.parquet")
    partitioner = RandomPartitioner(dataset, 128)
    for i in range(1,2):
        idx, partition = partitioner.partition()
        partition.to_parquet(f"openr1_128_shot_sample_{i}.parquet")
        with open(f"openr1_128_shot_sample_{i}.txt", "w") as f:
            for idx in idx:
                f.write(f"{idx}\n")