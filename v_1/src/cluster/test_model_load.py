"""
Test script: Load Qwen2.5-7B-Instruct on the cluster and verify:
1. Model loads and runs inference
2. We can extract hidden states (needed for Track B probing)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

print(f"Loading tokenizer for {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Loading model {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    output_hidden_states=True,  # needed for Track B activation extraction
)
print(f"Model loaded on: {model.device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

# --- Test 1: Simple inference ---
print("\n--- Test 1: Simple Inference ---")
prompt = "Translate this Akkadian text: šumma awīlum"
messages = [{"role": "user", "content": prompt}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Response: {response}")

# --- Test 2: Extract hidden states (what we need for Track B) ---
print("\n--- Test 2: Hidden State Extraction ---")
test_text = "ana šarri bēlīya aradka"  # sample Akkadian
inputs = tokenizer(test_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

hidden_states = outputs.hidden_states  # tuple of (num_layers + 1) tensors
print(f"Number of layers: {len(hidden_states) - 1}")  # -1 for embedding layer
print(f"Hidden state shape per layer: {hidden_states[0].shape}")
print(f"Hidden dimension: {hidden_states[0].shape[-1]}")

# Show we can get activations at any layer
for layer_idx in [0, len(hidden_states)//2, -1]:
    h = hidden_states[layer_idx]
    print(f"  Layer {layer_idx:>3}: mean={h.mean():.4f}, std={h.std():.4f}")

print("\nAll tests passed! Model is ready for Track B activation extraction.")
