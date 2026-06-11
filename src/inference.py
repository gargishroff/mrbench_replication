from __future__ import annotations
import os
import time
from typing import Protocol


class Backend(Protocol):
    name: str
    def generate(self, prompt: str, max_tokens: int = 64) -> str: ...


class HFBackend:

    def __init__(self, model_name: str, dtype: str = "bfloat16", device: str = "cuda"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, dtype),
            device_map=device,
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, max_tokens: int = 64) -> str:
        import torch
        messages = [{"role": "user", "content": prompt}]
        chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return full[len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):].strip()


class VLLMBackend:
    """vLLM in offline mode. Fast on a single GPU; batches internally."""

    def __init__(self, model_name: str, max_model_len: int = 4096, dtype: str = "bfloat16"):
        from vllm import LLM, SamplingParams
        self.name = model_name
        self.llm = LLM(model=model_name, max_model_len=max_model_len, dtype=dtype, gpu_memory_utilization=0.9)
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, prompt: str, max_tokens: int = 64) -> str:
        return self.generate_batch([prompt], max_tokens)[0]

    def generate_batch(self, prompts: list[str], max_tokens: int = 64) -> list[str]:
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        chats = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
        outputs = self.llm.generate(chats, sp)
        return [o.outputs[0].text.strip() for o in outputs]
    

def run_inference(df, backend: Backend, prompts_module, output_path: str, verbose: bool = True) -> None:
    import json
    from pathlib import Path

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    n_parse_fail = 0
    start = time.time()
    with out.open("w") as f:
        for idx, row in df.iterrows():
            prompt = prompts_module.make_prompt(row["history"], row["response"])
            try:
                raw = backend.generate(prompt, max_tokens=80)
                parsed = prompts_module.parse_response(raw)
            except Exception as exc:
                raw, parsed = f"ERROR: {exc}", None

            if parsed is None:
                n_parse_fail += 1

            f.write(json.dumps({
                "idx": int(idx),
                "conversation_id": row["conversation_id"],
                "tutor": row["tutor"],
                "raw": raw,
                "pred": parsed,
                "gold": {short: row[f"label_{short}"] for short in ("MI", "ML", "PG", "AC")},
            }) + "\n")
            n_done += 1
            if verbose and n_done % 50 == 0:
                elapsed = time.time() - start
                print(f"  {n_done}/{len(df)}  ({elapsed:.1f}s elapsed, {n_parse_fail} parse failures)")
    elapsed = time.time() - start
    print(f"Done: {n_done} responses in {elapsed:.1f}s; {n_parse_fail} parse failures")
