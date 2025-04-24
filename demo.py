from transformers import AutoTokenizer, BitNetForCausalLM, BitNetConfig
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Model identifier
model_id = "microsoft/bitnet-b1.58-2B-4T"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)

# charge explicitement la config BitNet pour ignorer l'auto_map cassé du repo
config = BitNetConfig.from_pretrained(model_id)

model = BitNetForCausalLM.from_pretrained(
    model_id,
    config=config,
    torch_dtype=torch.float32,     # dtype universel CPU/MPS
    low_cpu_mem_usage=False        # évite le dispatch Accelerate
).to("cpu")                        # charge et reste sur CPU

# Simple prompt
prompt = (
    "### Instruction:\n"
    "Réponds de façon concise et factuelle à la question suivante.\n\n"
    "### Question:\n"
    "Qui est Bill Gates ?\n\n"
    "### Réponse:"
)

# Tokenize and move tensors to the same device as the model
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate a response
outputs = model.generate(
    **inputs,
    max_new_tokens=120,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

# Decode and print the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
