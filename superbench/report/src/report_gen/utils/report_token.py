from transformers import GPT2Tokenizer

def ins_tokenizer():
    # use huggingface pretrained GPT2 tokenizer, BPE
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return tokenizer

def count_tokens(tokenizer, text):
    tokenized_text = tokenizer(text)["input_ids"]
    total_tokens = len(tokenized_text)
    return total_tokens