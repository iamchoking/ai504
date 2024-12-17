import os
import re
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from transformers import AutoTokenizer
from datasets import load_dataset

# Function to set random seed
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Function to Preprocess Data
class eli5_dataset(Dataset):
    def __init__(self,tokenizer, MAX_POSITION_EMBEDDINGS, data_type):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.block_size = MAX_POSITION_EMBEDDINGS
        
        if data_type == "train":
            data = load_dataset("eli5_category", split="train[:30000]", trust_remote_code=True)
            data = data.select(range(10000))
        elif data_type == "valid":
            data = load_dataset("eli5_category", split="validation1[:2000]", trust_remote_code=True)
        elif data_type == "test":
            data = load_dataset("eli5_category", split="test[:20]", trust_remote_code=True)

        data = data.flatten() 
        data = data.map(self.preprocess_function, batched=True,num_proc=8,remove_columns=data.column_names)
        data = data.map(self.group_texts, batched=True, num_proc=8)
        result =[]
        for i in data:
            result.append(i['input_ids'])
        self.final_data = torch.tensor(result).to(torch.int64)
        
    def preprocess_function(self, examples):
        return self.tokenizer([" ".join(x) for x in examples["answers.text"]])
    
    def group_texts(self, examples):

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]]) 
        if total_length >= (self.block_size-2):
            total_length = (total_length // (self.block_size-2)) * (self.block_size-2)
        result = {
            k: [[self.tokenizer.bos_token_id]+t[i : i + self.block_size-2]+[self.tokenizer.eos_token_id] for i in range(0, total_length, self.block_size-2)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
        
    def __len__(self):
        return len(self.final_data)
    
    def __getitem__(self, idx):
        return self.final_data[idx]


# Function to calculate perplexity
def calculate_perplexity(logits, targets):
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction='mean')
    perplexity = torch.exp(loss).item()
    return perplexity


# Main function to evaluate logits
def evaluate(logits_filename):
    set_seed(seed=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    BATCH_SIZE = 32
    MAX_POSITION_EMBEDDINGS = 200
    testset = eli5_dataset(tokenizer, MAX_POSITION_EMBEDDINGS, "test")
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE)
    print(f'Test dataset size: {len(test_loader.dataset)}')

    try:
        # Ensure file name is valid
        assert re.match(r"\d{8}\.npy", logits_filename), "File name must be an 8-digit student ID followed by '.npy'."

        # Load logits
        logits = np.load(logits_filename)
        logits = torch.from_numpy(logits)
        assert logits.shape == (len(test_loader.dataset), MAX_POSITION_EMBEDDINGS, 50257), f"Logits shape mismatch: expected ({len(test_loader.dataset)}, {MAX_POSITION_EMBEDDINGS}, 50257)."
        targets = torch.cat([target for target in test_loader]).cpu()
        
        # Calculate perplexity
        perplexity = calculate_perplexity(logits[:, :-1], targets[:, 1:])

    except AssertionError as e:
        perplexity = 1000
        print(f"Evaluation failed: {e}")

    print(f'{logits_filename[:-4]} - Perplexity: {round(perplexity)}')

if __name__ == "__main__":
    evaluate("20244512.npy")
    evaluate("20244513.npy")