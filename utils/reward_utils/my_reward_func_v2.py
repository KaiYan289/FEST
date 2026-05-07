import multiprocessing as mp
try:
    from ..openmathinst_utils import extract_answer, math_equal
except:
    from utils.openmathinst_utils import extract_answer, math_equal
import ray
from ray.exceptions import GetTimeoutError
from math_verify import verify, parse
from typing import Union

import signal
from contextlib import contextmanager

# 放在本文件顶层任意位置（确保在 my_reward_func 之前定义）

@ray.remote
def _omi_math_eval_worker_ray(solution_str: str, ground_truth: str):
    """
    在 Ray worker 进程中执行：
      - OMI: extract_answer + math_equal
      - math-verify: parse + verify
    返回 (omi_correct, mathv_correct)，都为 bool。
    """
    omi_correct = False
    mathv_correct = False

    # ---------- OMI 路径 ----------
    try:
        omi_pred = extract_answer(solution_str, extract_from_boxed=True)
        # math_equal 是同步函数，直接调用即可
        omi_correct = bool(
            math_equal(
                omi_pred,
                ground_truth,
                include_percentage=True,
                tolerance=1e-4,
                timeout=10.0,
                check_antlr_version=False,
            )
        )
    except Exception:
        omi_correct = False

    # ---------- math-verify 路径 ----------
    try:
        # 注意：把 parse 也放进 remote 里执行，避免本地 parse 卡死
        mathv_pred = parse(solution_str)
        gold_parsed = parse(f"\\boxed{{${ground_truth}$}}")
        mathv_correct = bool(
            verify(
                gold_parsed,
                mathv_pred,
                float_rounding=6,
                numeric_precision=15,
                strict=True,
                timeout_seconds=3,   # 这是 math-verify 自己的软超时
            )
        )
    except Exception:
        mathv_correct = False

    return omi_correct, mathv_correct


# ------------- 子进程里真正做 OMI + math_verify 的 worker ------------- #
def _math_eval_worker(solution_str: str, ground_truth: str, queue: "mp.Queue"):
    """
    在独立进程里做：
      - extract_answer + math_equal
      - parse + verify
    把 (omi_correct, mathv_correct) 通过队列传回主进程。
    无论中间报什么错，最终都保证 queue.put 一次，避免主进程永远阻塞。
    """
    omi_correct = False
    mathv_correct = False

    try:
        # ---- OMI path ----
        try:
            omi_pred = extract_answer(solution_str, extract_from_boxed=True)
            # math_equal 是同步函数，这里直接调用
            omi_correct = bool(math_equal(omi_pred, ground_truth, check_antlr_version=False))
        except Exception:
            omi_correct = False

        # ---- math-verify path ----
        try:
            mathv_pred = parse(solution_str)
            gold_parsed = parse(f"\\boxed{{${ground_truth}$}}")
            # 直接复用 verify 的调用方式，参数与你之前 verify_ray 一致
            mathv_correct = bool(
                verify(
                    gold_parsed,
                    mathv_pred,
                    float_rounding=6,
                    numeric_precision=15,
                    strict=True,
                    timeout_seconds=3,
                )
            )
        except Exception:
            mathv_correct = False
    finally:
        # 无论如何都尝试把结果塞回队列
        try:
            queue.put((omi_correct, mathv_correct))
        except Exception:
            # 即便队列出了问题，也不要再抛异常，交给父进程兜底
            pass


def my_reward_func(data_source, solution_str, ground_truth, extra_info) -> float:
    """
    兼容版 my_reward_func：
      - 函数签名不变
      - 返回结构不变：{"score": float, "acc": bool}
      - 内部把 OMI + math-verify 都丢到一个 Ray remote 任务里执行，
        用 ray.get(timeout=...) + ray.cancel(...) 兜底，防止卡死当前进程。
    """
    # 你可以按需调整这个 timeout（作用于整个 OMI+mathv 流程，而不是单个 verify）
    TOTAL_TIMEOUT_SECONDS = 10.0

    # 如果 Ray 还没 init（比如在本地调试脚本中调用），给一个纯 OMI 的安全回退
    if not ray.is_initialized():
        try:
            omi_pred = extract_answer(solution_str, extract_from_boxed=True)
            acc = bool(math_equal(omi_pred, ground_truth, check_antlr_version=False))
        except Exception:
            acc = False
        score = 1.0 if acc else 0.0
        return {"score": score, "acc": acc}

    # 正常情况：用 Ray remote worker 执行全部逻辑
    try:
        ref = _omi_math_eval_worker_ray.remote(solution_str, ground_truth)
    except Exception:
        # 如果 remote 调用本身失败，同样退回到纯 OMI 以保证不崩
        try:
            omi_pred = extract_answer(solution_str, extract_from_boxed=True)
            acc = bool(math_equal(omi_pred, ground_truth, check_antlr_version=False))
        except Exception:
            acc = False
        score = 1.0 if acc else 0.0
        return {"score": score, "acc": acc}

    try:
        omi_correct, mathv_correct = ray.get(ref, timeout=TOTAL_TIMEOUT_SECONDS)
    except GetTimeoutError:
        # 整个任务超时：强制取消这次评测，视为错误
        try:
            ray.cancel(ref, force=True)
        except Exception:
            pass
        omi_correct = False
        mathv_correct = False
    except Exception:
        # 任何其它异常也当做错误
        omi_correct = False
        mathv_correct = False

    acc = bool(omi_correct or mathv_correct)
    score = 1.0 if acc else 0.0

    return {
        "score": score,
        "acc": acc,
    }
