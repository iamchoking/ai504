import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from base import eli5_dataset, set_seed  # Importing the preprocessing and seed-setting functions

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, max_seq_length, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_length, d_model)
        )
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="relu"
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Add token and positional embeddings
        seq_length = x.size(1)
        x = self.token_embedding(x) + self.positional_encoding[:, :seq_length, :]
        for layer in self.layers:
            # Pass tgt as both tgt and memory to avoid the need for a separate encoder
            x = layer(tgt=x, memory=x)
        x = self.norm(x)
        return self.output_layer(x)

def truncate_and_split(batch, max_length):
    """Ensure all sequences are truncated or split into chunks of max_length."""
    # Check if batch is a tensor or dictionary
    if isinstance(batch, dict):
        batch = batch["input_ids"]  # Extract the tensor from the dictionary

    # Truncate and split sequences longer than max_length
    truncated_batch = []
    for sequence in batch:
        for i in range(0, len(sequence), max_length):
            chunk = sequence[i:i + max_length]
            truncated_batch.append(chunk)
    # Pad sequences to max_length for uniform shape
    padded_batch = torch.zeros((len(truncated_batch), max_length), dtype=torch.long)
    for i, chunk in enumerate(truncated_batch):
        padded_batch[i, :len(chunk)] = chunk  # Avoid unnecessary tensor conversion
    return padded_batch

def train_model(model, train_loader, optimizer, criterion, num_epochs, device, max_length):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            # Truncate and split sequences if necessary
            batch = truncate_and_split(batch, max_length).to(device)
            optimizer.zero_grad()
            logits = model(batch[:, :-1])
            
            # Use .reshape instead of .view to avoid memory stride issues
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return model

def save_logits(model, test_loader, filename, device, max_length):
    model.eval()
    logits = []
    with torch.no_grad():
        for batch in test_loader:
            # Truncate and split sequences if necessary
            batch = truncate_and_split(batch, max_length).to(device)
            # output = model(batch[:, :-1])  # Exclude the last token
            output = model(batch)
            logits.append(output.cpu())  # Collect logits
    # Concatenate logits along the batch dimension
    logits = torch.cat(logits, dim=0)
    
    # Ensure the shape matches (75, 200, 50257)
    expected_shape = (75, 200, 50257)
    assert logits.shape == expected_shape, f"Logits shape mismatch: expected {expected_shape}, got {logits.shape}"
    
    # Save logits as numpy array
    np.save(filename, logits.numpy())

def main():
    set_seed(seed=0)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    BATCH_SIZE = 32
    MAX_POSITION_EMBEDDINGS = 200
    VOCAB_SIZE = tokenizer.vocab_size
    NUM_LAYERS = 6
    D_MODEL = 512
    N_HEADS = 8
    D_FF = 2048
    DROPOUT = 0.1
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-4

    trainset = eli5_dataset(tokenizer, MAX_POSITION_EMBEDDINGS, "train")
    validset = eli5_dataset(tokenizer, MAX_POSITION_EMBEDDINGS, "valid")
    testset = eli5_dataset(tokenizer, MAX_POSITION_EMBEDDINGS, "test")

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(validset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_seq_length=MAX_POSITION_EMBEDDINGS,
        dropout=DROPOUT
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    model = train_model(model, train_loader, optimizer, criterion, NUM_EPOCHS, device, MAX_POSITION_EMBEDDINGS)

    print("Saving logits...")
    save_logits(model, test_loader, "20244512.npy", device, MAX_POSITION_EMBEDDINGS)
    print("Logits saved as 20244512.npy")

if __name__ == "__main__":
    main()
