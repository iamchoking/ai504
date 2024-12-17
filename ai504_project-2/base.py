import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer


########################## MUST NOT MODIFY #########################
# Function to set random seed
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Function for preprocessing data
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
#######################################################################
    

def main():
    ########################## MUST NOT MODIFY #########################
    # Set seed once at the top of your code (before executing any other code)
    set_seed(seed=0)
    
    # Preprocessing code
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    BATCH_SIZE = 32
    MAX_POSITION_EMBEDDINGS = 200
    trainset = eli5_dataset(tokenizer, MAX_POSITION_EMBEDDINGS, "train")
    validset = eli5_dataset(tokenizer, MAX_POSITION_EMBEDDINGS, "valid")
    testset = eli5_dataset(tokenizer, MAX_POSITION_EMBEDDINGS, "test")
    
    print(len(trainset)) # 17655
    print(len(validset)) # 5344
    print(len(testset)) # 75
    print(len(trainset[0])) # 200
    print(len(validset[0])) # 200
    print(len(testset[0])) # 200
    
    # Must use the provided train_dataloader, valid_dataloader, test_dataloader
    train_dataloader = DataLoader(trainset, batch_size=BATCH_SIZE)
    valid_dataloader = DataLoader(validset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(testset, batch_size=BATCH_SIZE)
    #######################################################################
    