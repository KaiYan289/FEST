import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F
import torch.nn.functional as F
from typing import Optional, Union

def masked_min(
    x: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[Union[int, tuple[int, ...]]] = None,
    keepdim: bool = False,
    default: Optional[float] = None,
) -> torch.Tensor:
    """
    masked_min similar to masked_mean:
      - only entries where mask==True are considered
      - if all masked along dim, return `default` (or +inf if default is None)

    Args:
        x: Tensor
        mask: bool/binary Tensor broadcastable to x
        dim: reduce dim(s). If None, reduce all elements.
        keepdim: keep reduced dim(s)
        default: value to return when all mask are False along reduced dims.
                 If None, returns +inf (or dtype max) for those positions.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()

    # Broadcast mask to x if needed
    mask = mask.expand_as(x)

    # fill invalid positions with +inf (dtype-aware)
    if x.dtype.is_floating_point:
        fill = torch.tensor(torch.finfo(x.dtype).max, device=x.device, dtype=x.dtype)
    else:
        # for integer tensors, use max value
        fill = torch.tensor(torch.iinfo(x.dtype).max, device=x.device, dtype=x.dtype)

    x_masked = x.masked_fill(~mask, fill)

    if dim is None:
        # reduce all
        m = x_masked.min()
        if default is None:
            return m
        # if no valid entries at all
        has_any = mask.any()
        return torch.where(has_any, m, torch.tensor(default, device=x.device, dtype=x.dtype))
    else:
        m = x_masked.min(dim=dim, keepdim=keepdim).values
        if default is None:
            return m

        # detect "all masked" along dim(s)
        has_any = mask.any(dim=dim, keepdim=keepdim)
        default_t = torch.tensor(default, device=x.device, dtype=x.dtype)
        return torch.where(has_any, m, default_t)

def masked_max(
    x: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[Union[int, tuple[int, ...]]] = None,
    keepdim: bool = False,
    default: Optional[float] = None,
) -> torch.Tensor:
    """
    masked_max similar to masked_mean:
      - only entries where mask==True are considered
      - if all masked along dim, return `default` (or +inf if default is None)

    Args:
        x: Tensor
        mask: bool/binary Tensor broadcastable to x
        dim: reduce dim(s). If None, reduce all elements.
        keepdim: keep reduced dim(s)
        default: value to return when all mask are False along reduced dims.
                 If None, returns +inf (or dtype max) for those positions.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()

    # Broadcast mask to x if needed
    mask = mask.expand_as(x)

    # fill invalid positions with +inf (dtype-aware)
    if x.dtype.is_floating_point:
        fill = torch.tensor(torch.finfo(x.dtype).min, device=x.device, dtype=x.dtype)
    else:
        # for integer tensors, use max value
        fill = torch.tensor(torch.iinfo(x.dtype).min, device=x.device, dtype=x.dtype)

    x_masked = x.masked_fill(~mask, fill)

    if dim is None:
        # reduce all
        m = x_masked.max()
        if default is None:
            return m
        # if no valid entries at all
        has_any = mask.any()
        return torch.where(has_any, m, torch.tensor(default, device=x.device, dtype=x.dtype))
    else:
        m = x_masked.max(dim=dim, keepdim=keepdim).values
        if default is None:
            return m

        # detect "all masked" along dim(s)
        has_any = mask.any(dim=dim, keepdim=keepdim)
        default_t = torch.tensor(default, device=x.device, dtype=x.dtype)
        return torch.where(has_any, m, default_t)


def compute_v3_loss(
    ref_logprob,
    ref_gold_logprob,
    grpo_old_pi_logprob,
    batch_pi_logprob,
    gold_logprob,
    batch_response_mask,
    gold_response_mask,
    beta_gw,
    beta_gl,
    beta_wl,
    on_mask,
    win_mask,
    advantages,
    dpo_loss_ratio,
    mode,
    grpo_clip_flag=False,
    dpo_margin=0.0,
    dpo_clamp_value=-50.0,
    token_flag='mean',
    sft_weight_for_weighted=-1,
    rl_weight_for_weighted=-1,
    offline_adv_for_luffy=None,
    luffy_rollout_n=8,
    for_debug_name=None,
    epsilon_1=0.2,
    epsilon_2=0.3,
):
    print(f"[debug] mode: {mode} token_flag: {token_flag} grpo_clip_flag: {grpo_clip_flag} dpo_clamp_value: {dpo_clamp_value} dpo_margin: {dpo_margin}")
    # gold is the preferred in DPO; win is the one in batch_pi with reward 1; lose is the one in batch_pi with reward 0
    # beta_gw: for those with on_mask = 1 and win_mask = 1 (avg reward for this question in this batch > 0, and this entry reward = 1);
    # beta_gl: for those with on_mask = 0 (avg reward for this question in this batch = 0. This entry reward must be 0); 
    # beta_wl: for those with on_mask = 1 and win_mask = 0 (avg reward for this question in this batch > 0, and this entry reward = 0)
    M, N = batch_pi_logprob.shape[0], batch_pi_logprob.shape[-1]
    if win_mask is not None and on_mask is not None:
        assert (win_mask * (1 - on_mask)).sum() == 0, "NOT win_mask = 0 wherever on_mask = 0!"
    if on_mask is not None and advantages is not None:
        assert (advantages.sum(dim=-1) * (1 - on_mask)).abs().max().item() == 0, "NOT adv_gold = 0 whenever on_mask = 0!" # assert advantage = 0 whenever on_mask = 0

    if mode in ['v3', 'v3_nogoldrl', 'v3_dposft', 'v3_smartsft', 'v3_smartsftrl', 'v3_smartrl']:
        lp, gp = (batch_pi_logprob * batch_response_mask).sum(dim=-1), (gold_logprob * gold_response_mask).sum(dim=-1)
        lp_ref, gp_ref = (ref_logprob * batch_response_mask).sum(dim=-1), (ref_gold_logprob * gold_response_mask).sum(dim=-1)
        r_pi = lp - lp_ref
        r_gold = gp - gp_ref
        delta = (r_gold - r_pi - dpo_margin).clamp_min(dpo_clamp_value) # this is deprecated in our experiment and always set to (almost) minus infinity for our method and variants/ablations (starts with "v3")
        z_gw = beta_gw * delta
        z_gl = beta_gl * delta
        z_wl = beta_wl * delta

        with torch.no_grad():
            z_gw_mean = verl_F.masked_mean(z_gw, win_mask * on_mask)
            z_gl_mean = verl_F.masked_mean(z_gl, (1 - on_mask))
            z_wl_mean = verl_F.masked_mean(z_wl, (1 - win_mask) * on_mask)
            z_gw_min = masked_min(z_gw, win_mask * on_mask)
            z_gl_min = masked_min(z_gl, (1 - on_mask))
            z_wl_min = masked_min(z_wl, (1 - win_mask) * on_mask)

        loss_gw = verl_F.masked_sum(-F.logsigmoid(z_gw), win_mask * on_mask)
        loss_gl = verl_F.masked_sum(-F.logsigmoid(z_gl), (1 - on_mask))
        loss_wl = verl_F.masked_sum(-F.logsigmoid(z_wl), (1 - win_mask) * on_mask)
        loss_dpo = dpo_loss_ratio * (loss_gw + loss_gl + loss_wl) / M
        
        if for_debug_name is not None:
            import os
            import time
            import uuid
            # save delta, beta_gw, beta_gl, beta_wl, win_mask, on_mask to a file {for_debug_name}.pt
            debug_payload = {
                # core dpo quantities
                "delta": delta.detach().cpu(),
                "z_gw": z_gw.detach().cpu(),
                "z_gl": z_gl.detach().cpu(),
                "z_wl": z_wl.detach().cpu(),

                # rewards / logprob summaries
                "lp": lp.detach().cpu(),
                "gp": gp.detach().cpu(),
                "lp_ref": lp_ref.detach().cpu(),
                "gp_ref": gp_ref.detach().cpu(),
                "r_pi": r_pi.detach().cpu(),
                "r_gold": r_gold.detach().cpu(),

                # masks / labels
                "win_mask": win_mask.detach().cpu() if win_mask is not None else None,
                "on_mask": on_mask.detach().cpu() if on_mask is not None else None,
                "advantages": advantages.detach().cpu() if advantages is not None else None,
                "batch_response_mask": batch_response_mask.detach().cpu(),
                "gold_response_mask": gold_response_mask.detach().cpu(),

                # hyperparameters / metadata
                "beta_gw": beta_gw,
                "beta_gl": beta_gl,
                "beta_wl": beta_wl,
                "dpo_loss_ratio": dpo_loss_ratio,
                "dpo_margin": dpo_margin,
                "dpo_clamp_value": dpo_clamp_value,
                "mode": mode,
                "token_flag": token_flag,
                "grpo_clip_flag": grpo_clip_flag,
                "sft_weight_for_weighted": sft_weight_for_weighted,
                "rl_weight_for_weighted": rl_weight_for_weighted,
            }
            ts_ns = time.time_ns()
            unique_id = uuid.uuid4().hex 
            filename = f"debug_{ts_ns}_{unique_id}.pt"
            save_path = "histogram/" + for_debug_name + "/" + filename

            save_dir = os.path.dirname(save_path)
            if save_dir:
               os.makedirs(save_dir, exist_ok=True)
               with open(save_path, "wb") as f:
                   torch.save(debug_payload, f)
                   f.flush()

            print(f"[debug] saved debug payload to {save_path}")

    if mode in ['HPT', 'HPT_nogoldrl', 'weighted', 'weighted_nogoldrl']:
        if mode in ['HPT', 'HPT_nogoldrl']:
            print("[debug] mode: simulating HPT")
            assert grpo_clip_flag == False and token_flag == 'sum', "HPT uses token sum and no clipping!"
            sft_loss = 0.3 * compute_sft_loss(gold_logprob, gold_response_mask * (1 - on_mask).unsqueeze(-1))['sft_loss'] 
            if mode != 'HPT_nogoldrl':
                pg_return = compute_grpo_loss(log_prob=batch_pi_logprob, old_log_prob=grpo_old_pi_logprob, eos_mask=batch_response_mask * on_mask.unsqueeze(-1), advantages=advantages, grpo_clip_flag=grpo_clip_flag, token_flag=token_flag, epsilon_1=epsilon_1, epsilon_2=epsilon_2)
                pg_loss, ratio_mean, ratio_max, ratio_min, ratio_clip_fraction = pg_return['loss'], pg_return['ratio_mean'], pg_return['ratio_max'], pg_return['ratio_min'], pg_return['ratio_clip_fraction']
                loss = pg_loss + sft_loss
            else:
                loss = sft_loss
        elif mode in ['weighted', 'weighted_nogoldrl']:
            print("[debug] mode: naive weighted RL+SFT on gold data, vanilla RL on unlabeled data")
            sft_loss = sft_weight_for_weighted * compute_sft_loss(gold_logprob, gold_response_mask)['sft_loss'] 
            if mode != 'weighted_nogoldrl':
                pg_return = compute_grpo_loss(log_prob=batch_pi_logprob, old_log_prob=grpo_old_pi_logprob, eos_mask=batch_response_mask, advantages=advantages, grpo_clip_flag=grpo_clip_flag, token_flag=token_flag, epsilon_1=epsilon_1, epsilon_2=epsilon_2)
                pg_loss, ratio_mean, ratio_max, ratio_min, ratio_clip_fraction = pg_return['loss'], pg_return['ratio_mean'], pg_return['ratio_max'], pg_return['ratio_min'], pg_return['ratio_clip_fraction']
                loss = pg_loss + sft_loss
            else:
                loss = sft_loss
        return_dict =  {"loss": loss, "loss_sft": sft_loss.detach().mean().item()}

        if mode not in ['HPT_nogoldrl', 'weighted_nogoldrl']:
            return_dict.update({
                "loss_rl": pg_loss.detach().mean().item(),
                "ratio_mean": ratio_mean,
                "ratio_max": ratio_max,
                "ratio_min": ratio_min,
                "ratio_clip_fraction": ratio_clip_fraction,
            })
        return return_dict
    elif mode == 'CHORD': # CHORD-phi; weighted SFT
        logp = torch.exp(gold_logprob)
        phi = (logp * (1 - logp)).detach()
        phi = phi.to(device=gold_logprob.device, dtype=gold_logprob.dtype)
        sft_loss = compute_sft_loss(gold_logprob * phi, gold_response_mask)['sft_loss']
        loss = sft_loss + 0.0 * batch_pi_logprob.sum() # to activate hook to adjust precision
        print(f"[debug] loss:{loss} phi: {phi.shape} {phi.dtype} logp: {logp.shape} {logp.dtype}")
        return_dict = {
            "loss": sft_loss,
            "phi_mean": phi.detach().mean().item(),
            "phi_top1pct": torch.quantile(phi * gold_response_mask, 0.99).detach().item()
        }
        return return_dict
        
    elif mode in ['ReLIFT', 'RL']: # just RL loss on gold data
        pg_return = compute_grpo_loss(log_prob=batch_pi_logprob, old_log_prob=grpo_old_pi_logprob, eos_mask=batch_response_mask, advantages=advantages, grpo_clip_flag=grpo_clip_flag, token_flag=token_flag, epsilon_1=epsilon_1, epsilon_2=epsilon_2)
        pg_loss, ratio_mean, ratio_max, ratio_min, ratio_clip_fraction = pg_return['loss'], pg_return['ratio_mean'], pg_return['ratio_max'], pg_return['ratio_min'], pg_return['ratio_clip_fraction']
        return_dict = {
            "loss": pg_loss,
            "ratio_mean": ratio_mean,
            "ratio_max": ratio_max,
            "ratio_min": ratio_min,
            "ratio_clip_fraction": ratio_clip_fraction,
        }
        return return_dict
    elif mode == 'LUFFY': 
        N = luffy_rollout_n
        new_advantages_rl, new_advantages_gold = advantages, offline_adv_for_luffy.unsqueeze(-1) # calculate new_advantage
        # on policy part
        pg_return = compute_grpo_loss(log_prob=batch_pi_logprob, old_log_prob=grpo_old_pi_logprob, eos_mask=batch_response_mask, advantages=new_advantages_rl, grpo_clip_flag=grpo_clip_flag, token_flag=token_flag, epsilon_1=epsilon_1, epsilon_2=epsilon_2)
        pg_loss, ratio_mean, ratio_max, ratio_min, ratio_clip_fraction = pg_return['loss'], pg_return['ratio_mean'], pg_return['ratio_max'], pg_return['ratio_min'], pg_return['ratio_clip_fraction']
        # off policy part
        exped_log_prob = torch.exp(gold_logprob)
        normalized_log_prob = exped_log_prob / (exped_log_prob + 0.1)
        pg_loss2 = verl_F.masked_mean(-normalized_log_prob * new_advantages_gold, gold_response_mask) if token_flag == 'mean' else verl_F.masked_sum(-normalized_log_prob * new_advantages_gold, gold_response_mask) / batch_response_mask.shape[-1]
        return_dict = {
            "loss": pg_loss * (N / (N + 1)) + pg_loss2 * (1 / (N + 1)),
            "normalized_log_prob": normalized_log_prob.detach().mean().item(),
            "on_policy_loss": pg_loss.item(),
            "off_policy_loss": pg_loss2.item(),
            "ratio_mean": ratio_mean,
            "ratio_max": ratio_max,
            "ratio_min": ratio_min,
            "ratio_clip_fraction": ratio_clip_fraction
        }
        return return_dict
    else:
        assert mode in ['v3', 'v3_nogoldrl', 'v3_dposft', 'v3_smartsft', 'v3_smartsftrl', 'v3_smartrl'], "mode Error!"
        assert grpo_clip_flag == True and token_flag == 'sum', "ours uses sum and clipping!"
        
        print("[debug] mode: ours with GRPO")
        if mode not in ['v3_smartsft', 'v3_smartsftrl', 'v3_smartrl']:
            loss = loss_dpo
        if mode == 'v3':
            pg_return = compute_grpo_loss(log_prob=batch_pi_logprob, old_log_prob=grpo_old_pi_logprob, eos_mask=batch_response_mask, advantages=advantages, grpo_clip_flag=grpo_clip_flag, token_flag=token_flag, epsilon_1=epsilon_1, epsilon_2=epsilon_2)
            pg_loss, ratio_mean, ratio_max, ratio_min, ratio_clip_fraction = pg_return['loss'], pg_return['ratio_mean'], pg_return['ratio_max'], pg_return['ratio_min'], pg_return['ratio_clip_fraction']
            loss += pg_loss
        elif mode == 'v3_dposft':
            sft_loss = compute_sft_loss(gold_logprob, gold_response_mask)['sft_loss']
            assert sft_weight_for_weighted > 0.0, "sft_weight_for_weighted must be > 0.0 for v3_dposft!"
            loss += sft_weight_for_weighted * sft_loss
        elif mode == 'v3_smartsft':
            gw_weight = F.sigmoid(-z_gw).detach() * win_mask * on_mask 
            gl_weight = F.sigmoid(-z_gl).detach() * (1 - on_mask)
            wl_weight = F.sigmoid(-z_wl).detach() * (1 - win_mask) * on_mask
            
            weight = gw_weight + gl_weight + wl_weight
            loss = verl_F.masked_mean(-gold_logprob * weight.unsqueeze(-1), gold_response_mask)
            
        elif mode == 'v3_smartsftrl':
            gw_weight = F.sigmoid(-z_gw).detach() * win_mask * on_mask 
            gl_weight = F.sigmoid(-z_gl).detach() * (1 - on_mask)
            wl_weight = F.sigmoid(-z_wl).detach() * (1 - win_mask) * on_mask
            
            weight = (gw_weight + gl_weight + wl_weight).unsqueeze(-1).repeat(1, gold_logprob.shape[-1])
            loss_sft = verl_F.masked_mean(-gold_logprob * weight, gold_response_mask)
            pg_return = compute_grpo_loss(log_prob=batch_pi_logprob, old_log_prob=grpo_old_pi_logprob, eos_mask=batch_response_mask, advantages=-weight, grpo_clip_flag=grpo_clip_flag, token_flag=token_flag, epsilon_1=epsilon_1, epsilon_2=epsilon_2)
            pg_loss, ratio_mean, ratio_max, ratio_min, ratio_clip_fraction = pg_return['loss'], pg_return['ratio_mean'], pg_return['ratio_max'], pg_return['ratio_min'], pg_return['ratio_clip_fraction']
            loss = loss_sft + pg_loss 
        elif mode == 'v3_smartrl':
            gw_weight = F.sigmoid(-z_gw).detach() * win_mask * on_mask 
            gl_weight = F.sigmoid(-z_gl).detach() * (1 - on_mask)
            wl_weight = F.sigmoid(-z_wl).detach() * (1 - win_mask) * on_mask
            
            weight = (gw_weight + gl_weight + wl_weight).unsqueeze(-1).repeat(1, gold_logprob.shape[-1])
            pg_return = compute_grpo_loss(log_prob=batch_pi_logprob, old_log_prob=grpo_old_pi_logprob, eos_mask=batch_response_mask, advantages=-weight, grpo_clip_flag=grpo_clip_flag, token_flag=token_flag, epsilon_1=epsilon_1, epsilon_2=epsilon_2)
            pg_loss, ratio_mean, ratio_max, ratio_min, ratio_clip_fraction = pg_return['loss'], pg_return['ratio_mean'], pg_return['ratio_max'], pg_return['ratio_min'], pg_return['ratio_clip_fraction']
            loss = pg_loss 
            
        return_dict = {
            "loss": loss,
            "loss_dpo": loss_dpo.detach().mean().item(),
            "loss_gw": loss_gw.detach().mean().item(),
            "loss_gl": loss_gl.detach().mean().item(),
            "loss_wl": loss_wl.detach().mean().item(),
            "delta_mean": delta.detach().mean().item(),
            "delta_min": delta.detach().min().item(),
            "z_wl_mean": z_wl.detach().mean().item(),
            "z_wl_min": z_wl.detach().min().item(),
            "z_gl_mean": z_gl.detach().mean().item(),
            "z_gl_min":  z_gl.detach().min().item(),
            "z_gw_mean": z_gw.detach().mean().item(),
            "z_gw_min": z_gw.detach().min().item(),
        }
        if mode == 'v3':
            return_dict.update({
                "loss_rl": pg_loss.detach().mean().item(),
                "ratio_mean": ratio_mean,
                "ratio_max": ratio_max,
                "ratio_min": ratio_min,
                "ratio_clip_fraction": ratio_clip_fraction,
            })
        elif mode == 'v3_dposft':
            return_dict.update({
                "loss_sft": sft_loss.detach().mean().item(),
                "sft_weight": sft_weight_for_weighted,
            })
        elif mode == 'v3_smartsft':
            del return_dict['loss_dpo']
            return_dict.update({
                "weight_gw": verl_F.masked_mean(gw_weight, win_mask * on_mask).detach().item(),
                "weight_gl": verl_F.masked_mean(gl_weight, 1 - on_mask).detach().item(),
                "weight_wl": verl_F.masked_mean(wl_weight, 1 - win_mask).detach().item(),
                "count_gw": (win_mask * on_mask).sum().detach().item(),
                "count_gl": (1 - on_mask).sum().detach().item(),
                "count_wl": ((1 - win_mask) * on_mask).sum().detach().item() 
            })
        elif mode == 'v3_smartrl':
            del return_dict['loss_dpo']
            return_dict.update({
                'pg_loss': pg_loss.detach().mean().item(), 
                "weight_gw": verl_F.masked_mean(gw_weight, win_mask * on_mask).detach().item(),
                "weight_gl": verl_F.masked_mean(gl_weight, 1 - on_mask).detach().item(),
                "weight_wl": verl_F.masked_mean(wl_weight, 1 - win_mask).detach().item(),
                "count_gw": (win_mask * on_mask).sum().detach().item(),
                "count_gl": (1 - on_mask).sum().detach().item(),
                "count_wl": ((1 - win_mask) * on_mask).sum().detach().item() 
            })
        elif mode == 'v3_smartsftrl':
            del return_dict['loss_dpo']
            return_dict.update({
                'sft_loss': loss_sft.detach().mean().item(),
                'pg_loss': pg_loss.detach().mean().item(), 
                "weight_gw": verl_F.masked_mean(gw_weight, win_mask * on_mask).detach().item(),
                "weight_gl": verl_F.masked_mean(gl_weight, 1 - on_mask).detach().item(),
                "weight_wl": verl_F.masked_mean(wl_weight, 1 - win_mask).detach().item(),
                "count_gw": (win_mask * on_mask).sum().detach().item(),
                "count_gl": (1 - on_mask).sum().detach().item(),
                "count_wl": ((1 - win_mask) * on_mask).sum().detach().item() 
            })
        return return_dict

def compute_grpo_loss(log_prob, old_log_prob, eos_mask, advantages, grpo_clip_flag=False, token_flag='mean', return_mid=False, epsilon_1=0.2, epsilon_2=0.3):
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    if grpo_clip_flag:
        ratio_clipped = torch.clamp(ratio, 1.0 - epsilon_1, 1.0 + epsilon_2)
        pg_loss_tokens = torch.max(-ratio * advantages, -ratio_clipped * advantages) 
        if token_flag == 'mean':
            pg_loss = verl_F.masked_mean(pg_loss_tokens, eos_mask)
        else:
            pg_loss = verl_F.masked_sum(pg_loss_tokens, eos_mask) / eos_mask.shape[-1]
        if return_mid:
            return pg_loss_tokens, eos_mask
    else:
        if token_flag == 'mean':
            pg_loss = verl_F.masked_mean(-ratio * advantages, eos_mask)
        else:
            pg_loss = verl_F.masked_sum(-ratio * advantages, eos_mask) / eos_mask.shape[-1]
        if return_mid:
            return -ratio * advantages, eos_mask
    return_dict = {
        "loss": pg_loss,
        "ratio_mean": verl_F.masked_mean(ratio, eos_mask).detach().item(),
        "ratio_max": masked_max(ratio, eos_mask).detach().item(),
        "ratio_min": masked_min(ratio, eos_mask).detach().item(),
        "ratio_clip_fraction": 0.0,
    }
    if grpo_clip_flag:
        return_dict["ratio_clip_fraction"] = verl_F.masked_mean((ratio_clipped != ratio).float(), eos_mask).detach().item()
    return return_dict

def compute_sft_loss(log_prob, eos_mask):
    sft_loss = verl_F.masked_mean(-log_prob, eos_mask)
    return {
        "sft_loss": sft_loss,
    }   
