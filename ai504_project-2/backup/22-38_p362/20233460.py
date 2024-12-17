import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import math
from base import eli5_dataset, set_seed  # Assuming eli5_dataset and set_seed are custom

# Hyperparameters
BATCH_SIZE = 32
MAX_POSITION_EMBEDDINGS = 200
LEARNING_RATE = 5e-5
NUM_EPOCHS = 5
VOCAB_SIZE = 50257  # GPT-2 vocabulary size
GRADIENT_ACCUMULATION_STEPS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for reproducibility
set_seed(seed=0)

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)

# Update the tokenizer padding token (GPT-2 doesn't have padding by default)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
train_dataset = eli5_dataset(tokenizer, MAX_POSITION_EMBEDDINGS, "train")
val_dataset = eli5_dataset(tokenizer, MAX_POSITION_EMBEDDINGS, "valid")
test_dataset = eli5_dataset(tokenizer, MAX_POSITION_EMBEDDINGS, "test")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize optimizer, scheduler, and criterion
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

# Mixed precision scaler
scaler = GradScaler()

# Training function
def train_model(model, train_loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for step, batch in enumerate(train_loader):
            # Move data to device
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            optimizer.zero_grad()

            # Mixed-precision training
            with autocast():
                outputs = model(inputs, labels=targets)
                loss = outputs.loss
            
            # Backpropagation
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            scaler.step(optimizer)
            scaler.update()
            
            # Update scheduler
            scheduler.step()

            # Track loss
            epoch_loss += loss.item()

            # Gradient accumulation
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = epoch_loss / len(train_loader)
        val_perplexity = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Training Loss: {avg_loss}, Validation Perplexity: {val_perplexity}")

# Validation function
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            outputs = model(inputs, labels=targets)
            loss = outputs.loss
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    perplexity = math.exp(avg_loss)  # Calculate perplexity
    return perplexity

# Logits generation function for test set
def generate_logits(model, test_loader, device):
    model.eval()
    logits_list = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch.to(device)
            outputs = model(inputs)
            logits_list.append(outputs.logits.cpu().numpy())
    logits_array = np.concatenate(logits_list, axis=0)
    return logits_array

# Main function to orchestrate training, evaluation, and saving results
def main():
    # Train the model
    train_model(model, train_loader, optimizer, scheduler, criterion=nn.CrossEntropyLoss(), device=DEVICE)

    # Generate logits for the test set
    logits = generate_logits(model, test_loader, DEVICE)

    # Save logits to a .npy file
    np.save("20233460.npy", logits)

if __name__ == "__main__":
    main()
