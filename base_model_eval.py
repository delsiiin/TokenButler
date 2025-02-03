import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import tqdm

def evaluate_wikitext2(model, tokenizer, max_seq_len=512):
    """
    Evaluates the model on the Wikitext-2 dataset using perplexity.

    Args:
        model: The neural network model.
        tokenizer: The tokenizer corresponding to the model.
        max_seq_len: Maximum sequence length for tokenization.

    Returns:
        The perplexity of the model on the test dataset.
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    concatenated_text = "\n\n".join(dataset["text"])
    tokenized_output = tokenizer(concatenated_text, truncation=False)

    input_ids = torch.tensor(tokenized_output["input_ids"]).to(device)
    num_chunks = (len(input_ids) + max_seq_len - 1) // max_seq_len
    input_chunks = input_ids.split(max_seq_len)

    loss_fn = CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0
    progress_bar = tqdm.tqdm(range(num_chunks), desc="Evaluating Wikitext-2")

    for chunk in progress_bar:
        batch = input_chunks[chunk].unsqueeze(0).to(device)
        if batch.size(1) < 2:
            continue
        with torch.no_grad():
            outputs = model(batch, use_cache=False)
            logits = outputs.logits[:, :-1, :]
            target = batch[:, 1:]

            loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))
            total_loss += loss.item() * target.numel()
            total_tokens += target.numel()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"Perplexity on Wikitext-2: {perplexity.item()}")
    return perplexity.item()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Wikitext-2 perplexity.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model')
    args = parser.parse_args()

    model_path = args.model_path
    print(f"Loading model: {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    perplexity = evaluate_wikitext2(model, tokenizer, max_seq_len=4096)
