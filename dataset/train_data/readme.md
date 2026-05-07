Note: 

Run prepare_train.py to generate the RL-only dataset. 

64, 128, 256, and 512 shot data under sample_1 are uniformly randomly sampled from the 46K full dataset. 128 shot with sample_2 and sample_3 are distinct subsets within the 512 shot dataset. 

plot_jsonl_output_token_histograms.py is a debugging script that counts rollout lengths and plot histograms.

"HPT format" is essentially the same as a normal dataset (Elliott/Openr1-Math-46k-8192 on huggingface), but without system prompt (following HPT).