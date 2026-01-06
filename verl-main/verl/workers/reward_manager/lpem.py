# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict

import torch
import re
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from transformers import AutoTokenizer


TAG   = re.compile(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", re.I | re.S)
BOX   = re.compile(r"(?:\\{1,2}\(|)"          # 允许 \(... 或 \(  可省略
                   r"\\{1,2}boxed\s*\{\s*([^}]*)\s*}"  # \boxed 或 \\boxed
                   r"(?:\)|)", re.S)
INLINE = re.compile(r"\$([^$]+)\$|\\\(([^)]+)\\\)", re.S)
FRAC  = re.compile(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)")
NUM   = re.compile(r"-?\d+(?:\.\d+)?")

MEMORY_RE = re.compile(r"<memory_item>(.*?)</memory_item>", re.DOTALL)

def format_pass(resp: str) -> bool:
    """能提取到 <memory_item> 且内容非空视为格式 OK"""
    m = MEMORY_RE.search(resp)
    return bool(m and m.group(1).strip())

def _norm(num: str) -> str:
    """把 '3.0' → '3'，'-4/2' → '-2' 之类"""
    try:
        f = float(num)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return num.strip()

def sanitize_math_answer(txt: str) -> str:
    txt = txt.strip()

    # 1) <answer> 标签
    if m := TAG.search(txt):
        txt = m.group(1).strip()

    # 2) \boxed{...}
    elif m := BOX.search(txt):
        txt = m.group(1).strip()

    # 3) $...$ or \( ... \)
    elif m := INLINE.search(txt):
        txt = (m.group(1) or m.group(2)).strip()

    # 4) 处理分数 p/q  或  \frac{p}{q}
    #    先把 \frac{p}{q} 变成 p/q
    txt = re.sub(r"\\frac\s*\{\s*([^}]+?)\s*}\s*\{\s*([^}]+?)\s*}", r"\1/\2", txt)

    if m := FRAC.search(txt):
        p, q = map(float, m.groups())
        if q:                                          # 避免除 0
            return _norm(str(p / q))

    # 5) 最后再抓单个数字
    if m := NUM.search(txt):
        return _norm(m.group())

    # 都没抓到就返回原始（去空白）
    return txt

def normalize_number_format(x: str) -> str:
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return x

@register("lpem_reward")
class LPEMRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        rm_tokenizer_path: str | None = None,

    ) -> None:
        self.tokenizer = tokenizer
        self.rm_tokenizer = AutoTokenizer.from_pretrained(rm_tokenizer_path, use_fast=True, trust_remote_code=True)
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len        # Initialize rank attribute
        import torch.distributed

        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  #
            if self.rank == 0 and i == 0:
                print(f"data_item keys: {data_item.batch.keys()}")

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            # print("prompt_str:",prompt_str)
            # print("response_str:",response_str)

            is_format_ok = format_pass(response_str)
            # reward_extra_info["format_ok"].append(is_format_ok)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            # print("ground_truth:",ground_truth)

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            # print("data_source:",data_source)
            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            responses_grm = data_item.batch.get("responses_grm", [])
            # print("total keys:", data_item.batch.keys())
            result = None
            if responses_grm is not None and len(responses_grm) > 0:
                if not isinstance(responses_grm[0], str):
                    responses_grm = self.rm_tokenizer.decode(responses_grm, skip_special_tokens=True)
                    # print("#######################")
                    # print("responses_grm:",responses_grm)
                    # print("#######################")
                # Judgment: Correct / Incorrect
                ans = normalize_number_format(sanitize_math_answer(responses_grm.strip()))
                gt_ = normalize_number_format(sanitize_math_answer(ground_truth))
                correct = 0.0
                if ans == gt_:
                    correct = 1.0
                else:
                    correct = 0.0

                if correct and is_format_ok:
                    score = 1.0
                else:
                    score = 0.0 
                    
                result = {
                    "score": score,
                    "acc": 1.0 if correct > 0 else 0.0,  # acc should be 1.0 for correct, 0.0 for incorrect
                    "format": is_format_ok,
                    "pred": responses_grm,

                }
            else:
                if self.rank == 0:
                    print("No valid responses_grm found, falling back to compute_score")

            if result is None:
                result = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            reward = score

            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor