import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torchvision.models import resnet50
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

# Constants (from previous implementation)
CIFAR_BATCH_SIZE = 128
LM_BATCH_SIZE = 32
VL_BATCH_SIZE = 16
MAX_LENGTH = 128
HIDDEN_SIZE = 768
NUM_EPOCHS = 2
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

# Reuse previous dataset classes from the last implementation
# (ELI5Dataset, LLaVADataset, get_cifar10_loaders)

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

# Vision Encoder: Enhanced ResNet with Projection Layer
class VisionEncoder(nn.Module):
    def __init__(self, num_img_tokens=32):
        super().__init__()
        # Use ResNet50 as backbone
        resnet = resnet50(pretrained=True)
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x, mode='vision'):
        # Extract features
        features = self.features(x)
        
        # Global average pooling
        pooled = self.pool(features).flatten(1)
        
        if mode == 'classification':
            # Use classification head
            return self.classification_head(pooled)
        
        # Return raw features for vision-language model
        return pooled

# Text Decoder: Enhanced GPT-2
class TextDecoder(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        # Use pre-trained GPT-2 model
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Resize token embeddings to include special tokens
        self.model.resize_token_embeddings(len(tokenizer))
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )

# Vision-Language Model
class VisionLanguageModel(nn.Module):
    def __init__(self, vision_encoder, text_decoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        
        # Projection layer to align vision and text embeddings
        self.vision_projection = nn.Sequential(
            nn.Linear(2048, HIDDEN_SIZE),  # Adjust input dimension to match ResNet50 features
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE * NUM_IMG_TOKEN)
        )
    
    def forward(self, images, input_ids, attention_mask=None, labels=None):
        # Encode visual features using ResNet50 backbone
        visual_features = self.vision_encoder(images)
        
        # Project visual features
        batch_size = images.size(0)
        projected_visual_features = self.vision_projection(visual_features)
        
        # Reshape projected features
        visual_tokens = projected_visual_features.view(batch_size, NUM_IMG_TOKEN, HIDDEN_SIZE)
        
        # Prepare input for text decoder
        decoder_input_ids = input_ids.clone()
        
        # Compute loss
        outputs = self.text_decoder(
            input_ids=decoder_input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
        
        return outputs

def train_vision_encoder(model, dataloader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Compute classification logits
            logits = model(images, mode='classification')
            
            # Compute loss
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Vision Encoder - Epoch {epoch+1}/{NUM_EPOCHS}, Avg Loss: {avg_loss:.4f}")
    
    return model

def train_text_decoder(model, dataloader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Text Decoder - Epoch {epoch+1}/{NUM_EPOCHS}, Avg Loss: {avg_loss:.4f}")
    
    return model

def train_vision_language_model(model, dataloader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images, input_ids, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Vision-Language Model - Epoch {epoch+1}/{NUM_EPOCHS}, Avg Loss: {avg_loss:.4f}")
    
    return model

def generate_test_logits(model, test_loader, device):
    model.eval()
    all_logits = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            
            # Get logits from text decoder
            outputs = model.text_decoder(input_ids)
            logits = outputs.logits
            
            all_logits.append(logits.cpu().numpy())
    
    # Concatenate and slice to match required shape
    all_logits = np.concatenate(all_logits, axis=0)
    
    # Ensure shape is (20, 32, 50257)
    all_logits = all_logits[:20, :32, :50257]
    
    return all_logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default="./instruct_tuning/instruct.json")
    parser.add_argument('--image_folder_path', type=str, default="./instruct_tuning/images/")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(IMG_PATCH, special_tokens=True)

    # Prepare datasets
    cifar_trainloader, _ = get_cifar10_loaders()
    eli5_dataset = ELI5Dataset(tokenizer, MAX_LENGTH, 'train')
    eli5_loader = DataLoader(eli5_dataset, batch_size=LM_BATCH_SIZE, shuffle=True)
    llava_dataset = LLaVADataset(args.json_path, args.image_folder_path, tokenizer, is_train=True)
    llava_loader = DataLoader(llava_dataset, batch_size=VL_BATCH_SIZE, shuffle=True)
    test_llava_dataset = LLaVADataset(args.json_path, args.image_folder_path, tokenizer, is_train=False)
    test_llava_loader = DataLoader(test_llava_dataset, batch_size=VL_BATCH_SIZE, shuffle=False)

    # Initialize models
    vision_encoder = VisionEncoder().to(device)
    text_decoder = TextDecoder(tokenizer).to(device)
    vision_language_model = VisionLanguageModel(vision_encoder, text_decoder).to(device)

    # Training stages
    print("Training Vision Encoder...")
    vision_encoder = train_vision_encoder(vision_encoder, cifar_trainloader, device)

    print("Training Text Decoder...")
    text_decoder = train_text_decoder(text_decoder, eli5_loader, device)

    print("Training Vision-Language Model...")
    vision_language_model = train_vision_language_model(vision_language_model, llava_loader, device)

    # Generate logits
    print("Generating Test Logits...")
    test_logits = generate_test_logits(vision_language_model, test_llava_loader, device)

    # Save logits
    np.save('20244512.npy', test_logits)
    print("Logits saved successfully!")

if __name__ == "__main__":
    main()