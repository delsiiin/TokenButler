from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# Reasoning-based math question
question = "If a train leaves New York at 2 PM traveling 60 mph and another train leaves Los Angeles at 3 PM traveling 80 mph on the same track towards each other, when will they meet?"

# Tokenize input
inputs = tokenizer(question, return_tensors="pt").to("cuda")

# Generate response
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)

# Decode and print output
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
