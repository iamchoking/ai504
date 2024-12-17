import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from PIL import Image
import json
from datasets import load_dataset
import copy
import os
import torch.nn.functional as F
import argparse
import random

import torch.nn as nn
import torch.optim as optim

############################# UNCHANGED #############################

# Constants
CIFAR_BATCH_SIZE = 128
LM_BATCH_SIZE = 32
VL_BATCH_SIZE = 16
MAX_LENGTH = 128
HIDDEN_SIZE = 768
NUM_EPOCHS = 1
IMG_PATCH = '<img>'
NUM_IMG_TOKEN = 32
VLM_MAX_LENGTH = 32

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# CIFAR-10 Dataset and DataLoader
def get_cifar10_loaders():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=CIFAR_BATCH_SIZE,
                           shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=CIFAR_BATCH_SIZE,
                          shuffle=False, num_workers=2)
    
    return trainloader, testloader

# ELI5 Dataset
class ELI5Dataset(Dataset):
    def __init__(self,tokenizer, MAX_POSITION_EMBEDDINGS, data_type):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.block_size = MAX_POSITION_EMBEDDINGS
        
        if data_type == "train":
            data = load_dataset("eli5_category", split="train[:3000]", trust_remote_code=True)
            data = data.select(range(1000))
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

# LLaVA Dataset
def transform_fn(is_train):
    if is_train:
        return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    else:
        return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Do not change
class LLaVADataset(Dataset):
    def __init__(self, json_file, img_path, tokenizer, is_train):
        super().__init__()

        self.transform = transform_fn(is_train)

        self.json_file = json_file

        self.tokenizer = tokenizer
        self.img_path = img_path

        self.ignore_idx = -100
        self.begin_signal = tokenizer.bos_token
        self.end_signal = tokenizer.eos_token

        with open(self.json_file) as json_file:
            data = json.load(json_file)

        if is_train:
            data = data[:1000]
        else:
            data = data[1000:]

        self.data = data

    def preprocess(self, conversation):
        question = self.begin_signal + "human: " + conversation[0]['value'] + self.end_signal
        answer = self.begin_signal + "assistant: " + conversation[1]['value'] + self.end_signal

        tokenized_q = self.tokenizer(question, return_tensors="pt")

        combined_qa = question + answer
        tokenized_qa = self.tokenizer(combined_qa, padding="max_length", truncation=True,
                                      max_length=VLM_MAX_LENGTH, return_tensors="pt")

        input_ids = tokenized_qa.input_ids[0]
        label = copy.deepcopy(input_ids)
        len_of_q = len(tokenized_q.input_ids[0])
        label[:len_of_q] = self.ignore_idx

        len_of_pad = tokenized_qa.input_ids.eq(self.tokenizer.pad_token_id).sum().item()
        label[-len_of_pad:] = self.ignore_idx

        return input_ids, label
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        meta = self.data[idx]

        image_id = meta['image']
        image = Image.open(os.path.join(self.img_path, image_id)).convert('RGB')
        image = self.transform(image)

        conversation = meta['conversation']
        input_id, label = self.preprocess(conversation)

        return dict(image=image, input_ids=input_id, label=label)
    
############################# /UNCHANGED #############################]
'''
# Vision Encoder Model (ResNet18)
class VisionEncoder(nn.Module):
    def __init__(self):
        super(VisionEncoder, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, HIDDEN_SIZE)

    def forward(self, x):
        return self.resnet18(x)

# Text Decoder Model (GPT-2)
class TextDecoder(nn.Module):
    def __init__(self):
        super(TextDecoder, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')

    def forward(self, input_ids, attention_mask=None):
        return self.gpt2(input_ids, attention_mask=attention_mask, labels=input_ids)

# Vision-Language Model
class VisionLanguageModel(nn.Module):
    def __init__(self, vision_encoder, text_decoder):
        super(VisionLanguageModel, self).__init__()
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.img_token = nn.Parameter(torch.randn(NUM_IMG_TOKEN, HIDDEN_SIZE))
        self.linear_projection = nn.Linear(HIDDEN_SIZE, self.text_decoder.gpt2.config.n_embd)

    def forward(self, images, input_ids):
        image_features = self.vision_encoder(images)
        img_token_embeddings = self.linear_projection(image_features)
        img_token_embeddings = img_token_embeddings.unsqueeze(1).expand(-1, NUM_IMG_TOKEN, -1)

        # Concatenate image tokens and text tokens
        inputs = torch.cat((img_token_embeddings, input_ids), dim=1)
        return self.text_decoder(inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str)
    parser.add_argument('--image_folder_path', type=str)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(IMG_PATCH, special_tokens=True)

    # cifar10
    cifar_trainloader, cifar_testloader = get_cifar10_loaders()
    
    # eli5
    eli5_dataset = ELI5Dataset(tokenizer, MAX_LENGTH, 'train')
    eli5_loader = DataLoader(eli5_dataset, batch_size=LM_BATCH_SIZE, shuffle=True)
    
    # llava train
    llava_dataset = LLaVADataset(args.json_path, args.image_folder_path, tokenizer, is_train=True)
    llava_loader = DataLoader(llava_dataset, batch_size=VL_BATCH_SIZE, shuffle=True)
    
    # llava test
    test_llava_dataset = LLaVADataset(args.json_path, args.image_folder_path, tokenizer, is_train=False)
    test_llava_loader = DataLoader(test_llava_dataset, batch_size=VL_BATCH_SIZE, shuffle=False)
    
    
    # Model
    vision_encoder = VisionEncoder()
    text_decoder = TextDecoder()
    model = VisionLanguageModel(vision_encoder, text_decoder)
    model.to(device)

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    # Training
    for epoch in range(NUM_EPOCHS):
        model.train()
        for idx, (images, texts) in enumerate(zip(trainloader, elify_train_dataset)):
            images = images[0].to(device)
            input_ids = texts['input_ids'].to(device)

            optimizer.zero_grad()
            outputs = model(images, input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Step {idx}/{len(trainloader)}, Loss: {loss.item()}")

    # Save logits
    logits = outputs.logits.detach().cpu().numpy()
    np.save('20244512.npy', logits)


if __name__ == "__main__":
    main()
'''

# Import necessary packages
from transformers import GPT2LMHeadModel, GPT2Config
import torch.optim as optim
import torch.nn as nn

# Step 1: Vision Encoder (using ResNet18)
class VisionEncoder(nn.Module):
    def __init__(self):
        super(VisionEncoder, self).__init__()
        # Load pretrained ResNet18 model and remove the final fully connected layer
        resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove last fully connected layer
        self.fc = nn.Linear(resnet.fc.in_features, HIDDEN_SIZE)  # Map to the hidden size of GPT-2

    def forward(self, x):
        x = self.resnet(x)  # Pass image through ResNet
        x = torch.flatten(x, 1)  # Flatten the output for the linear layer
        x = self.fc(x)  # Project to the desired hidden size
        return x

# Step 2: Text Decoder (GPT-2 model)
class TextDecoder(nn.Module):
    def __init__(self):
        super(TextDecoder, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

# Step 3: Combine Vision Encoder and Text Decoder into Vision-Language Model
class VisionLanguageModel(nn.Module):
    def __init__(self):
        super(VisionLanguageModel, self).__init__()
        self.vision_encoder = VisionEncoder()
        self.text_decoder = TextDecoder()
        self.linear = nn.Linear(HIDDEN_SIZE, self.text_decoder.model.config.n_embd)  # Mapping vision to text embedding space

    def forward(self, images, input_ids, attention_mask=None, labels=None):
        visual_features = self.vision_encoder(images)  # Get visual features from the encoder
        visual_features = self.linear(visual_features)  # Map to the text input space
        output = self.text_decoder(input_ids, attention_mask=attention_mask, labels=labels)  # Pass through the text decoder
        return output

# Training loop for Vision-Language Model
def train_vlm(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass through the vision-language model
        outputs = model(images, input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# Function to save logits as a .npy file
def save_logits(model, dataloader, filepath):
    model.eval()
    logits_list = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            outputs = model(images, input_ids)
            logits = outputs.logits
            logits_list.append(logits.cpu().numpy())
    
    logits_array = np.concatenate(logits_list, axis=0)
    np.save(filepath, logits_array)

# Main training routine
def main():
    # Set seed for reproducibility
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default="./instruct_tuning/instruct.json")
    parser.add_argument('--image_folder_path', type=str, default="./instruct_tuning/images/")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Load datasets
    # train_loader, _ = get_cifar10_loaders()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    train_dataset = LLaVADataset(args.json_path, args.image_folder_path, tokenizer, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=VL_BATCH_SIZE, shuffle=True)

    # Initialize model and optimizer
    model = VisionLanguageModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    # Training the model
    for epoch in range(NUM_EPOCHS):
        loss = train_vlm(model, train_dataloader, optimizer, device)
        print(f'Epoch {epoch + 1}, Loss: {loss}')

    # Save the logits to a .npy file
    save_logits(model, train_dataloader, '20244512.npy')

if __name__ == "__main__":
    main()