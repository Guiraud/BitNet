#!/usr/bin/env python3
"""
fast-demo.py – BitNet 2‑B quick‑start

Usage examples
--------------
# simple run (auto‑detects MPS if dispo)
uv run python fast-demo.py -p "Explique la théorie des jeux en 3 points"

# force CPU
uv run python fast-demo.py -d cpu -p "Bonjour"

Notes
-----
* 1st run: weights are unpacked + TorchInductor compiles kernels → 1‑2 min.
* Next runs reuse the on‑disk cache and answer in a few seconds.
"""

import argparse
import os
import time

import torch
from transformers import AutoTokenizer, BitNetConfig, BitNetForCausalLM

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "inductor-cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE_DIR
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable torch.compile / Inductor globally for stability
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
import torch._dynamo
torch._dynamo.disable()

# ▶︎ Prérequis macOS : brew install llvm libomp  (nécessaire pour l'OpenMP de torch.compile)
# Ensure clang++ with OpenMP (brew llvm/libomp) is on PATH for Inductor
LLVM_BIN = "/opt/homebrew/opt/llvm/bin"
if os.path.isdir(LLVM_BIN) and LLVM_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{LLVM_BIN}:{os.environ['PATH']}"
# Force Inductor to use that toolchain
os.environ.setdefault("CC", os.path.join(LLVM_BIN, "clang"))
os.environ.setdefault("CXX", os.path.join(LLVM_BIN, "clang++"))


def pick_device(user_choice: str | None = None) -> torch.device:
    """
    Return CPU by default. Use MPS only when explicitly requested with `-d mps`
    and when available. This avoids unstable Metal shader compilation.
    """
    if user_choice == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast BitNet demo")
    parser.add_argument(
        "-p", "--prompt", required=True, help="Prompt/question to ask the model"
    )
    parser.add_argument(
        "-d",
        "--device",
        choices=["cpu", "mps"],
        help="Force device (default: auto)",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        default=120,
        help="Maximum tokens to generate (default: 120)",
    )
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"▶ Loading BitNet on {device} …")
    if device.type == "cpu":
        print("ℹ️  Running in eager CPU mode (compile disabled).")
    elif device.type == "mps":
        print("ℹ️  Running on MPS in eager mode (compile disabled).")

    config = BitNetConfig.from_pretrained(MODEL_ID)
    dtype = torch.float16 if device.type == "mps" else torch.float32
    # If MPS is selected but compilation fails later, we will catch and retry on CPU.

    prompt = (
        "### Instruction:\n"
        "Réponds de façon concise et factuelle à la question suivante.\n\n"
        f"### Question:\n{args.prompt}\n\n"
        "### Réponse:"
    )

    try:
        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = BitNetForCausalLM.from_pretrained(
            MODEL_ID,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,  # avoid Accelerate dispatch
        ).to(device)
        load_time = time.perf_counter() - t0
        print(f"✅ Model ready in {load_time:,.1f}s")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        t0 = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        duration = time.perf_counter() - t0

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split("### Instruction")[0].strip()
        print("\n📝 Answer:\n")
        print(answer)
        print(f"\n⏱️ Generation time: {duration:,.1f}s")
    except Exception as e:
        if device.type == "mps":
            print(f"⚠️  MPS failed: {e}\n   → retrying on CPU…")
            device = torch.device("cpu")
            os.environ["TORCHINDUCTOR_DISABLE"] = "1"
            dtype = torch.float32
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = BitNetForCausalLM.from_pretrained(
                MODEL_ID, config=config, torch_dtype=dtype, low_cpu_mem_usage=False
            ).to(device)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = answer.split("### Instruction")[0].strip()
            print("\n📝 Answer (CPU fallback):\n")
            print(answer)
        else:
            raise
    finally:
        pass


if __name__ == "__main__":
    main()