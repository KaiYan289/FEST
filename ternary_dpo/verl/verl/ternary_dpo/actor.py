# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import math
import logging
import os
from typing import Tuple
from copy import deepcopy
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import numpy as np
from datetime import datetime

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def timeprint(*args, **kwargs):
    now = datetime.now().strftime("%m-%d-%Y %H:%M:%S")
    print(f"[{now}]", *args, **kwargs)

class TernaryDPODataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None, sft_actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.sft_actor_optimizer = sft_actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

        self.before_sft_grad_norm = None

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs
    def debug_param_grad_state(self, optimizer):
        for gi, group in enumerate(optimizer.param_groups):
            for pi, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                st = optimizer.state[p]
                dev_p, dtype_p = p.device, p.dtype
                dev_g, dtype_g = p.grad.device, p.grad.dtype

                # 先看 param vs grad
                if dev_p != dev_g or dtype_p != dtype_g:
                    print(
                        "[PG-MISMATCH]",
                        f"group {gi}, param {pi}:",
                        f"param: {dev_p}/{dtype_p}, grad: {dev_g}/{dtype_g}",
                    )
                    return
                else:
                    print(f"[PG-MATCH] group {gi}, param {pi}:", f"param: {dev_p}/{dtype_p}, grad: {dev_g}/{dtype_g}")

                for key in ["exp_avg", "exp_avg_sq"]:
                    if key in st:
                        t = st[key]
                        if t.device != dev_p or t.dtype != dtype_p:
                            print(
                                "[STATE-MISMATCH]",
                                f"group {gi}, param {pi}, key {key}:",
                                f"param: {dev_p}/{dtype_p}, state: {t.device}/{t.dtype}",
                            )
                            return
                        else:
                            print(f"[STATE-MATCH] group {gi}, param {pi}, key {key}:", f"param: {dev_p}/{dtype_p}, state: {t.device}/{t.dtype}")
        print("No param/grad/state mismatch for any param with grad.")

    def debug_group_devices(self, optimizer):
        for gi, group in enumerate(optimizer.param_groups):
            devs_param  = set()
            dtypes_param = set()
            devs_grad   = set()
            dtypes_grad  = set()

            for p in group["params"]:
                if p.grad is None:
                    continue
                devs_param.add(p.device)
                dtypes_param.add(p.dtype)
                devs_grad.add(p.grad.device)
                dtypes_grad.add(p.grad.dtype)

            print(f"[GROUP {gi}] param devices:", devs_param, "param dtypes:", dtypes_param)
            print(f"[GROUP {gi}] grad  devices:", devs_grad,  "grad  dtypes:", dtypes_grad)

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite or larger than max_grad_skip, skip the update
        if not torch.isfinite(grad_norm) or (grad_norm > self.config.max_grad_norm):
            print("RL grad set to zero")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm
    
    def _sft_optimizer_step(self):
        assert self.config.sft.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.sft.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.sft.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.sft.grad_clip)

        # if grad_norm is not finite, skip the update
        if (not torch.isfinite(grad_norm)) or (grad_norm > self.config.sft.max_sft_grad_norm) \
                or (self.config.sft.using_dynamic_max_sft_grad_norm and (self.before_sft_grad_norm is not None) and grad_norm > 1.5 * self.before_sft_grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite or too large: {grad_norm}")
            self.sft_actor_optimizer.zero_grad()
        else:
            self.before_sft_grad_norm = grad_norm
            self.sft_actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob_for_tensordict(self, data, max_token_len, calculate_entropy=True, use_dynamic_bsz=True, temperature=1.0) -> torch.Tensor:
        micro_batch_size = data['responses'].shape[0]
        if use_dynamic_bsz:
            micro_batches, indices = rearrange_micro_batches(batch=data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)
        
        log_probs_lst = []
        entropy_lst = []
        for idx, micro_batch in enumerate(micro_batches):
            timeprint("computing log prob for micro batch", idx, "/", len(micro_batches))
            entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]
            

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, batch: DataProto):
        raise NotImplementedError("placeholder!")


    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        
        self.actor_module.eval()


        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy_v3(self, rl_batch: DataProto, gold_batch: DataProto, gold_batch_for_sft_logprob: DataProto):
        data_to_return = { # debugging purposes; deprecated
            "binary_win_ref_logprobs": [],
            "binary_win_logprobs": [],
            "binary_win_masks": [],
            "binary_lose_ref_logprobs": [],
            "binary_lose_logprobs": [],
            "binary_lose_masks": [],
            "ternary_gold_ref_logprobs": [],
            "ternary_gold_logprobs": [],
            "ternary_gold_masks": [],
            "ternary_win_ref_logprobs": [],
            "ternary_win_logprobs": [],
            "ternary_win_masks": [],
            "ternary_lose_ref_logprobs": [],
            "ternary_lose_logprobs": [],
            "ternary_lose_masks": [],
        }
        self.actor_module.train()

        special_mode = self.config.special_mode
        temperature = rl_batch.meta_info["temperature"]
        current_global_step = rl_batch.meta_info["global_step"]
        dpo_loss_ratio = self.config.dpo_loss_ratio

        entropy_coeff = self.config.entropy_coeff
        loss_agg_mode = self.config.loss_agg_mode
        calculate_entropy = True # (entropy_coeff != 0)

        def get_response_mask(data: DataProto):
            responses = data["responses"]
            response_length = responses.size(1)
            attention_mask = data["attention_mask"]
            response_mask = attention_mask[:, -response_length:]
            return response_mask

        mb_size_pairs = self.config.ppo_mini_batch_size # actually twice the mini batch size - for every minibatch, we mix gradient from both batches
        micro_pairs_size = self.config.ppo_micro_batch_size_per_gpu
        metrics = {'temperature': temperature, 'global_step': current_global_step}
        # modes: ReLIFT, v3, HPT, weighted (naive weight between RL and SFT data), RL, SFT. ReLIFT here = RL

        from .core_algos import compute_sft_loss, compute_grpo_loss, compute_v3_loss

        grpo_clip_low = self.config.get("clip_ratio_low", self.config.get("clip_ratio", 0.2))
        grpo_clip_high = self.config.get("clip_ratio_high", self.config.get("clip_ratio", 0.3))

        if special_mode == 'SFT': # only go through SFT data
            sft_batch_dataloader = gold_batch_for_sft_logprob.batch.split(self.config.ppo_mini_batch_size)
            for epoch in range(self.config.ppo_epochs):
                for batch_idx, gold_batch in enumerate(sft_batch_dataloader):
                    gold_micro_batches = gold_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu # 16
                    self.actor_optimizer.zero_grad()
                    
                    for microbatch_idx, gold_micro_batch in enumerate(gold_micro_batches):
                        timeprint("micro batch count:", microbatch_idx, "/", len(gold_micro_batches), "minibatch count:", batch_idx, "/", len(sft_batch_dataloader))
                        if isinstance(gold_micro_batch, DataProto):
                            gold_micro_batch = {**gold_micro_batch.batch, **gold_micro_batch.non_tensor_batch}
                        else:
                            gold_micro_batch = gold_micro_batch.to(get_torch_device().current_device())

                        gold_batch_gold_response_mask = get_response_mask(gold_micro_batch)
                        gold_log_prob, entropy0 = self.compute_log_prob_for_tensordict(gold_micro_batch, max_token_len=self.config.ppo_max_token_len_per_gpu, calculate_entropy=calculate_entropy, use_dynamic_bsz=self.config.use_dynamic_bsz, temperature=temperature)
                        loss = compute_sft_loss(log_prob=gold_log_prob, eos_mask=gold_batch_gold_response_mask)['sft_loss']
                        entropy_loss = agg_loss(loss_mat=entropy0, loss_mask=gold_batch_gold_response_mask, loss_agg_mode=loss_agg_mode) * entropy_coeff
                        loss_old = loss.detach().item()
                        loss -= entropy_loss
                        loss = loss / self.gradient_accumulation
                        
                        res = {'sft_loss': loss_old, 'entropy_loss': entropy_loss.detach().item()}
                        loss.backward()
                    grad_norm = self._optimizer_step()
                    append_to_dict(metrics, {"grad_norm": grad_norm.detach().item()})
            self.actor_optimizer.zero_grad()
        else: # go through both data; need to mix both batches in one gradient step though
            for epoch in range(self.config.ppo_epochs):

                timeprint("[update policy v3] epoch:", epoch, "ternary stage begins")
                rl_weight, gold_weight = self.config.rl_data_weight, self.config.gold_data_weight
                rl_batch_dataloader = rl_batch.batch.split(self.config.ppo_mini_batch_size)

                assert len(gold_batch) >= len(rl_batch_dataloader), "insufficient gold data! This should not happen"
                assert len(gold_batch) == len(gold_batch_for_sft_logprob), "gold_batch_for_sft_logprob should have the same size as gold_batch!"

                rl_sizes, gold_data_size = [len(b) for b in rl_batch_dataloader], len(gold_batch)

                def allocate_gold_sizes_like_rl(rl_sizes, gold_total):
                    """
                    Given rl's mini-batch sizes (the last one might be smaller),
                    allocate gold_total to the same number of buckets, each >=1, and the sum == gold_total.
                    """
                    n = len(rl_sizes)
                    rl_sum = sum(rl_sizes)
                    exact = [gold_total * (s / rl_sum) for s in rl_sizes]
                    base = [int(math.floor(x)) for x in exact]
                    frac = [x - b for x, b in zip(exact, base)]

                    for i in range(n):
                        if base[i] < 1:
                            base[i] = 1
                            frac[i] = exact[i] - 1
                    
                    cur = sum(base)
                    if cur > gold_total:
                        need_remove = cur - gold_total
                        # 优先从 frac 小的减（它们本来就更接近 floor），并且只能从 base>1 的减
                        order = sorted(range(n), key=lambda i: frac[i])  # 小 frac 优先减
                        for i in order:
                            if need_remove == 0:
                                break
                            can = base[i] - 1
                            if can <= 0:
                                continue
                            dec = min(can, need_remove)
                            base[i] -= dec
                            need_remove -= dec
                        assert need_remove == 0, "Should be feasible given gold_total>=n"
                     # 如果 base 之和不足 gold_total：把剩余的加到 frac 大的桶里（largest remainder）
                    elif cur < gold_total:
                        need_add = gold_total - cur
                        order = sorted(range(n), key=lambda i: frac[i], reverse=True)  # 大 frac 优先加
                        # 循环分配，避免极端情况下只加在一个桶（其实也没问题，但循环更稳）
                        k = 0
                        while need_add > 0:
                            i = order[k % n]
                            base[i] += 1
                            need_add -= 1
                            k += 1
                    assert len(base) == n
                    assert all(x >= 1 for x in base)
                    assert sum(base) == gold_total
                    return base

                def split_dataproto_by_sizes(dp, sizes):
                    assert sum(sizes) == len(dp), "sizes must sum to the length of dp!"
                    datas, s = [], 0
                    for size in sizes:
                        datas.append(dp.slice(s, s+size).batch)
                        s += size
                    return datas

                gold_sizes = allocate_gold_sizes_like_rl(rl_sizes, gold_data_size)

                print("[debug] rl_sizes:", rl_sizes, "gold_data_size:", gold_data_size)
                print("[debug] gold_sizes:", gold_sizes)

                gold_batch_dataloader = split_dataproto_by_sizes(gold_batch, gold_sizes)   # gold_batch always have the same size as gold_batch_for_sft_logprob. In HPT they could be smaller than rl_batch.
                gold_batch_for_sft_logprob_dataloader = split_dataproto_by_sizes(gold_batch_for_sft_logprob, gold_sizes)
        
                for batch_idx, (rl_batch, gold_batch, gold_batch_for_sft_logprob) in enumerate(zip(rl_batch_dataloader, gold_batch_dataloader, gold_batch_for_sft_logprob_dataloader)):

                    rl_micro_batches, gold_micro_batches_gold, gold_micro_batches_for_sft_logprob = rl_batch.split(self.config.ppo_micro_batch_size_per_gpu), gold_batch.split(self.config.ppo_micro_batch_size_per_gpu), gold_batch_for_sft_logprob.split(self.config.ppo_micro_batch_size_per_gpu)
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu # 16 # 64 // 4. This is very important due to token sum that we are using
                    self.actor_optimizer.zero_grad()
                    timeprint("[debug-new] gradient-accumulation:", self.gradient_accumulation, "config-mini:", self.config.ppo_mini_batch_size, "config-micro:", self.config.ppo_micro_batch_size_per_gpu)
                    for microbatch_idx, (rl_micro_batch, gold_micro_batch_gold, gold_micro_batch_for_sft_logprob) in enumerate(zip(rl_micro_batches, gold_micro_batches_gold, gold_micro_batches_for_sft_logprob)):
                        timeprint("micro batch count:", microbatch_idx, "/", len(rl_micro_batches), "minibatch count:", batch_idx, "/", len(rl_batch_dataloader))
                        assert self.config.use_dynamic_bsz == True, "must use dynamic bsz for v3!"

                        if isinstance(rl_micro_batch, DataProto):
                            rl_micro_batch = {**rl_micro_batch.batch, **rl_micro_batch.non_tensor_batch}
                            gold_micro_batch_gold = {**gold_micro_batch_gold.batch, **gold_micro_batch_gold.non_tensor_batch}
                            gold_micro_batch_for_sft_logprob = {**gold_micro_batch_for_sft_logprob.batch, **gold_micro_batch_for_sft_logprob.non_tensor_batch}
                        else:
                            rl_micro_batch = rl_micro_batch.to(get_torch_device().current_device())
                            gold_micro_batch_gold = gold_micro_batch_gold.to(get_torch_device().current_device())
                            gold_micro_batch_for_sft_logprob = gold_micro_batch_for_sft_logprob.to(get_torch_device().current_device())
                        
                        rl_batch_response_mask = get_response_mask(rl_micro_batch)
                        gold_batch_gold_response_mask = get_response_mask(gold_micro_batch_gold)
                        gold_batch_for_sft_logprob_response_mask = get_response_mask(gold_micro_batch_for_sft_logprob)

                        if special_mode in ['v3', 'v3_nogoldrl', 'v3_dposft', 'v3_smartsft', 'v3_smartsftrl', 'v3_smartrl']:
                            gold_rollout_ref_prob, gold_label_data_ref_prob = gold_micro_batch_gold["ref_log_prob"].detach(), gold_micro_batch_for_sft_logprob["sft_ref_log_probs"].detach()
                        else:
                            gold_rollout_ref_prob, gold_label_data_ref_prob = None, None
                        rl_grpo_prob = rl_micro_batch["old_log_probs"].detach()
                        gold_grpo_prob = gold_micro_batch_gold["old_log_probs"].detach()

                        # first calculate RL part; this is just normal GRPO
                        if rl_weight != 0:
                            rl_log_prob, entropy0 = self.compute_log_prob_for_tensordict(rl_micro_batch, max_token_len=self.config.ppo_max_token_len_per_gpu, calculate_entropy=calculate_entropy, use_dynamic_bsz=self.config.use_dynamic_bsz, temperature=temperature)
                            rl_return_dict = compute_grpo_loss(log_prob=rl_log_prob, old_log_prob=rl_grpo_prob, eos_mask=rl_batch_response_mask, advantages=rl_micro_batch['advantages'].detach(), grpo_clip_flag=self.config.grpo_clip_flag, token_flag=self.config.token_flag, epsilon_1=grpo_clip_low, epsilon_2=grpo_clip_high)  
                            loss = rl_return_dict['loss']
                            entropy_loss = agg_loss(loss_mat=entropy0, loss_mask=rl_batch_response_mask, loss_agg_mode=loss_agg_mode) * entropy_coeff
                            loss -= entropy_loss
                            loss = rl_weight * loss / self.gradient_accumulation
                            
                            res = {'rl_part/grpo_loss': loss.detach().item(), 'rl_part/entropy_loss': entropy_loss.detach().item(), 'rl_part/ratio_mean': rl_return_dict['ratio_mean'], 'rl_part/ratio_max': rl_return_dict['ratio_max'], 'rl_part/ratio_min': rl_return_dict['ratio_min'], 'rl_part/ratio_clip_fraction': rl_return_dict['ratio_clip_fraction']}
                            append_to_dict(metrics, res)
                            del res
                            loss.backward()
                        else:
                            print("[debug] skipping RL batch due to zero weight...")

                        # then calculate DPO+RL / SFT+RL part for HPT, weighted, v3
                        if special_mode in ['HPT', 'HPT_nogoldrl', 'v3', 'v3_nogoldrl', 'weighted', 'weighted_nogoldrl', 'ReLIFT', 'RL', 'v3_dposft', 'v3_smartsft', 'v3_smartsftrl', 'v3_smartrl', 'LUFFY', 'CHORD']:
                            new_gold_rollout_batch_prob, entropy1 = self.compute_log_prob_for_tensordict(gold_micro_batch_gold, max_token_len=self.config.ppo_max_token_len_per_gpu, calculate_entropy=calculate_entropy, use_dynamic_bsz=self.config.use_dynamic_bsz, temperature=temperature)
                            new_gold_label_batch_prob, entropy2 = self.compute_log_prob_for_tensordict(gold_micro_batch_for_sft_logprob, max_token_len=self.config.ppo_max_token_len_per_gpu, calculate_entropy=calculate_entropy, use_dynamic_bsz=self.config.use_dynamic_bsz, temperature=temperature)

                            for_debug_name = self.config.get('for_debug_name', None)
                            if for_debug_name is not None:
                                from datetime import datetime
                                for_debug_name += datetime.now().strftime("%m-%d-%Y %H:%M:%S") 

                            luffy_n = int(self.config.get("rollout_n", 8))
                            outp = compute_v3_loss(
                                ref_logprob=gold_rollout_ref_prob,
                                ref_gold_logprob=gold_label_data_ref_prob,
                                grpo_old_pi_logprob=gold_grpo_prob,
                                batch_pi_logprob=new_gold_rollout_batch_prob,
                                gold_logprob=new_gold_label_batch_prob,
                                batch_response_mask=gold_batch_gold_response_mask,
                                gold_response_mask=gold_batch_for_sft_logprob_response_mask,
                                beta_gw = self.config.beta_gold_to_win,
                                beta_gl = self.config.beta_gold_to_lose,
                                beta_wl = self.config.beta_win_to_lose,
                                on_mask = gold_micro_batch_gold['on_mask'] if special_mode in ['HPT', 'HPT_nogoldrl', 'v3', 'v3_nogoldrl', 'v3_dposft', 'v3_smartsft', 'v3_smartsftrl', 'v3_smartrl'] else None,
                                win_mask = gold_micro_batch_gold['win_mask'] if special_mode in ['HPT', 'HPT_nogoldrl', 'v3', 'v3_nogoldrl', 'v3_dposft', 'v3_smartsft', 'v3_smartsftrl', 'v3_smartrl'] else None,
                                advantages = gold_micro_batch_gold['advantages'].detach(),
                                dpo_loss_ratio = dpo_loss_ratio,
                                dpo_margin=self.config.get("dpo_margin", 0),
                                mode = special_mode, # weighted, v3, HPT, relift, RL, weighted_nogoldrl, v3_nogoldrl, HPT_nogoldrl, relift_nogoldrl, RL_nogoldrl, v3_dposft, v3_smartsft, 'v3_smartsftrl', 'v3_smartrl', 'LUFFY'
                                grpo_clip_flag = self.config.grpo_clip_flag,
                                dpo_clamp_value = self.config.dpo_clamp_value,
                                token_flag = self.config.token_flag,
                                sft_weight_for_weighted = self.config.sft_weight_for_weighted,
                                rl_weight_for_weighted = self.config.rl_weight_for_weighted,
                                offline_adv_for_luffy = gold_micro_batch_gold['offline_adv'] if special_mode == 'LUFFY' else None,
                                luffy_rollout_n = luffy_n,
                                for_debug_name = self.config.get('for_debug_name', None),
                                epsilon_1=grpo_clip_low,
                                epsilon_2=grpo_clip_high,
                            )
                        
                            loss = outp['loss']
                        
                            if special_mode in  ['HPT', 'HPT_nogoldrl']:
                                print("[debug] simulate_hpt's entropy calculation")
                                entropy_loss = (verl_F.masked_sum(entropy1, gold_batch_gold_response_mask * gold_micro_batch_gold['on_mask'].unsqueeze(-1)) + verl_F.masked_sum(entropy2, gold_batch_for_sft_logprob_response_mask * (1 - gold_micro_batch_gold['on_mask']).unsqueeze(-1))) / ((gold_batch_gold_response_mask * gold_micro_batch_gold['on_mask'].unsqueeze(-1)).sum() + (gold_batch_for_sft_logprob_response_mask * (1 - gold_micro_batch_gold['on_mask']).unsqueeze(-1)).sum()) * entropy_coeff
                                # masked_mean of two parts with total token-mean
                            elif special_mode == 'LUFFY':
                                entropy_loss_pi, entropy_loss_gold = agg_loss(loss_mat=entropy1, loss_mask=gold_batch_gold_response_mask, loss_agg_mode=loss_agg_mode), agg_loss(loss_mat=entropy2, loss_mask=gold_batch_for_sft_logprob_response_mask, loss_agg_mode=loss_agg_mode)
                                n_l = float(luffy_n)
                                entropy_loss = (entropy_loss_pi * (n_l / (n_l + 1.0)) + entropy_loss_gold * (1.0 / (n_l + 1.0))) * entropy_coeff
                            elif special_mode == 'CHORD':
                                entropy_loss_pi, entropy_loss_gold = agg_loss(loss_mat=entropy1, loss_mask=gold_batch_gold_response_mask, loss_agg_mode=loss_agg_mode), agg_loss(loss_mat=entropy2, loss_mask=gold_batch_for_sft_logprob_response_mask, loss_agg_mode=loss_agg_mode)
                                entropy_loss = entropy_loss_gold * entropy_coeff + 0.0 * entropy_loss_pi # to activate hook for entropy1 to adjust precision
                            else:
                                entropy_loss_pi, entropy_loss_gold = agg_loss(loss_mat=entropy1, loss_mask=gold_batch_gold_response_mask, loss_agg_mode=loss_agg_mode), agg_loss(loss_mat=entropy2, loss_mask=gold_batch_for_sft_logprob_response_mask, loss_agg_mode=loss_agg_mode)
                                if special_mode.find("v3") != -1:
                                    entropy_loss = (entropy_loss_pi + dpo_loss_ratio * entropy_loss_gold) * entropy_coeff
                                elif special_mode.find("weighted") != -1:
                                    entropy_loss = (entropy_loss_pi * self.config.sft_weight_for_weighted + entropy_loss_gold * self.config.rl_weight_for_weighted) * entropy_coeff
                                else: # relift, rl
                                    entropy_loss = entropy_loss_pi * entropy_coeff
                                    
                            loss -= entropy_loss
                            loss = gold_weight * loss / self.gradient_accumulation
                            loss.backward()
                            new_outp = {'gold_part/entropy_loss': entropy_loss.detach().item()}
                            for k in outp.keys():
                                if k != 'loss':
                                    new_outp["gold_part/"+k] = outp[k]
                            append_to_dict(metrics, new_outp)
                            del outp
                        else:
                            assert special_mode in ['ReLIFT_nogoldrl', 'RL_nogoldrl'], 'mode Error!'
                            print(f"[debug] {special_mode} has no gold rl, skipping SFT/DPO on gold batch...")

                    grad_norm = self._optimizer_step()
                    append_to_dict(metrics, {"grad_norm": grad_norm.detach().item()})
                ########################
            self.actor_optimizer.zero_grad()

        timeprint(f"update_policy ends, {self.config.save_logprob_and_masks_for_debug}")
        if self.config.get('for_debug_name', None):
            exit(0)
        return metrics, data_to_return
    
    @GPUMemoryLogger(role="dp actor", logger=logger) # for ReLIFT
    def sft_update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()
        from .core_algos import compute_sft_loss
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.sft.sft_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.sft.sft_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.sft.sft_mini_batch_size // self.config.sft.sft_micro_batch_size_per_gpu # 16
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.sft.sft_micro_batch_size_per_gpu)

                self.sft_actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    entropy_coeff = self.config.sft.entropy_coeff

                    loss_type = self.config.sft.get("sft_loss_type", "v0")
                    
                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0 or loss_type in ["v1", "v3", "v4", "v5", "v6", "v4_per_sentence", "v5_per_sentence"]:
                        calculate_entropy = True
                    
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                    
                    if loss_type == "v0":
                        loss_fn = compute_sft_loss
                        ret_dict = loss_fn(log_prob=log_prob, eos_mask=response_mask)
                    else:
                        raise ValueError(f"Invalid sft loss type: {loss_type}")

                    sft_loss = ret_dict["sft_loss"]

                    if entropy_coeff != 0:
                        loss_agg_mode = self.config.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = sft_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = sft_loss
                        entropy_loss = 0

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.sft.sft_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation

                    loss.backward()

                    if entropy_coeff != 0:
                        data = {
                            'actor/combined_loss': float(policy_loss.detach().cpu().item()),
                            'actor/sft_loss': float(sft_loss.detach().cpu().item()),
                            'actor/sft_entropy_loss': float(entropy_loss.detach().cpu().item())
                        }
                    else:
                        data = {
                            'actor/combined_loss': float(policy_loss.detach().cpu().item()),
                            'actor/sft_loss': float(sft_loss.detach().cpu().item())
                        }
                    
                    append_to_dict(metrics, data)

                grad_norm = self._sft_optimizer_step()
                data = {"actor/sft_grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
                
        self.sft_actor_optimizer.zero_grad()
        return metrics
    
