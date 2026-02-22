import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import DataHandler


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))




class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        
        # Key, Query, Value projections combined
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        # Causal mask: lower triangular matrix
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dim (n_embd)
        
        # Compute queries, keys, and values for all heads in batch
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # each is (B, T, C)
        
        # Split into heads: (B, T, C) -> (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        
        # Compute attention scores: (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Apply causal mask: fill upper triangular with -inf
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax over last dimension
        att = F.softmax(att, dim=-1)
        
        # Apply attention to values: (B, n_head, T, head_dim)
        y = att @ v
        
        # Re-assemble heads: (B, n_head, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.c_proj(y)
        
        return y 
        

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_head, n_embd, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model """


    def __init__(self, n_layer, n_head, n_embd, vocab_size, block_size):
        super().__init__()

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, self.n_embd),
            wpe = nn.Embedding(block_size, self.n_embd),            
            h = nn.ModuleList([Block(n_head, n_embd, block_size) for _ in range(self.n_layer)]),
            ln_f = nn.LayerNorm(self.n_embd),
        ))
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)



    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits





def generate_text(model, data_handler, start_text, num_chars, block_size, device, top_k=None):
    """
    Generate text using the model.
    
    Args:
        model: The GPT model
        data_handler: DataHandler for encoding/decoding
        start_text: Starting text for generation
        num_chars: Number of characters to generate
        block_size: Maximum context length
        device: Device to run on
        top_k: If specified, only sample from top k tokens
    
    Returns:
        Generated text string
    """
    model.eval()
    generated = start_text
    
    with torch.no_grad():
        for _ in range(num_chars):
            # Encode the context (last block_size characters)
            context = generated[-block_size:] if len(generated) >= block_size else generated
            tokens = torch.tensor(data_handler.encoder(context), dtype=torch.long).unsqueeze(0).to(device)
            
            # Get model predictions
            logits = model(tokens)
            # Take logits for the last position
            logits = logits[:, -1, :]  # (1, vocab_size)
            
            if top_k is not None:
                # Top-k sampling: zero out all but top k logits
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Decode and append
            next_char = data_handler.decoder(next_token[0].tolist())
            generated += next_char
    
    return generated


def compute_accuracy(logits, targets):
    """
    Compute accuracy: ratio of correctly predicted last characters.
    
    Args:
        logits: Model output (B, T, vocab_size)
        targets: Target tokens (B, T)
    
    Returns:
        Accuracy as a float
    """
    # Get predictions for the last position
    predictions = logits[:, -1, :].argmax(dim=-1)  # (B,)
    targets_last = targets[:, -1]  # (B,)
    correct = (predictions == targets_last).float().sum()
    accuracy = correct / targets_last.size(0)
    return accuracy.item()


def train_model(
        train_path,
        test_path=None,
        model=None,                        
        block_size=10,
        n_layer=3,
        n_head=3,
        n_embd=48,
        learning_rate=3e-4,
        batch_size=64,
        epochs=10
):            
    np.random.seed(42)
    torch.manual_seed(42)
    
    data_handler = DataHandler(train_path, test_path, block_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = data_handler.get_vocab_size()
    if model is None:
        model = GPT(n_layer, n_head, n_embd, vocab_size, block_size)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()
    print('Using device:', device)

    trainset = data_handler.get_dataset('train')
    testset = data_handler.get_dataset('test')
    
    # setup the dataloader
    train_loader = DataLoader(
        trainset,
        sampler=torch.utils.data.RandomSampler(trainset, replacement=True, num_samples=int(1e5)),
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,        
    )     
    if testset:       
        test_loader = DataLoader(
            testset,
            sampler=torch.utils.data.RandomSampler(testset, replacement=False, num_samples=int(1e4)),
            shuffle=False,
            pin_memory=True,
            batch_size=batch_size,            
        )
    
    # Track metrics
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    # Store generated sentences
    generated_sentences_regular = []
    generated_sentences_topk = []

    for ep in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {ep + 1}/{epochs}")
        print('='*60)
        
        # Training
        model.train()
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        num_train_batches = 0
        
        for i, batch in enumerate(tqdm(train_loader, desc="Training")):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(x)  # (B, T, vocab_size)
            
            # Reshape for cross entropy: (B*T, vocab_size) vs (B*T,)
            B, T, V = logits.shape
            loss = criterion(logits.view(B*T, V), y.view(B*T))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_train_loss += loss.item()
            epoch_train_acc += compute_accuracy(logits, y)
            num_train_batches += 1
        
        avg_train_loss = epoch_train_loss / num_train_batches
        avg_train_acc = epoch_train_acc / num_train_batches
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        
        # Evaluation
        model.eval()
        epoch_test_loss = 0.0
        epoch_test_acc = 0.0
        num_test_batches = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                
                logits = model(x)
                
                B, T, V = logits.shape
                loss = criterion(logits.view(B*T, V), y.view(B*T))
                
                epoch_test_loss += loss.item()
                epoch_test_acc += compute_accuracy(logits, y)
                num_test_batches += 1
        
        avg_test_loss = epoch_test_loss / num_test_batches
        avg_test_acc = epoch_test_acc / num_test_batches
        test_losses.append(avg_test_loss)
        test_accs.append(avg_test_acc)
        
        print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")
        
        # Generate sentences (regular sampling)
        print("\nGenerated sentences (regular sampling):")
        epoch_sentences_regular = []
        for j in range(3):
            sentence = generate_text(model, data_handler, "the ", 30, block_size, device, top_k=None)
            print(f"  {j+1}: {sentence}")
            epoch_sentences_regular.append(sentence)
        generated_sentences_regular.append(epoch_sentences_regular)
        
        # Generate sentences (top-k sampling with k=5)
        print("\nGenerated sentences (top-k=5 sampling):")
        epoch_sentences_topk = []
        for j in range(3):
            sentence = generate_text(model, data_handler, "the ", 30, block_size, device, top_k=5)
            print(f"  {j+1}: {sentence}")
            epoch_sentences_topk.append(sentence)
        generated_sentences_topk.append(epoch_sentences_topk)
    
    # ============================================================
    # Plot training curves
    # ============================================================
    import os
    os.makedirs("plots", exist_ok=True)
    
    epochs_range = range(1, epochs + 1)
    
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, test_losses, label='Test Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label='Train Accuracy', marker='o')
    plt.plot(epochs_range, test_accs, label='Test Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/transformer_training_curves.png", dpi=150)
    plt.close()
    print("\nSaved: plots/transformer_training_curves.png")
    
    # ============================================================
    # Summary of generated sentences
    # ============================================================
    print("\n" + "="*60)
    print("SUMMARY: Generated Sentences Comparison")
    print("="*60)
    
    print("\n--- Regular Sampling ---")
    for ep_idx, sentences in enumerate(generated_sentences_regular):
        print(f"Epoch {ep_idx + 1}:")
        for s in sentences:
            print(f"  {s}")
    
    print("\n--- Top-k=5 Sampling ---")
    for ep_idx, sentences in enumerate(generated_sentences_topk):
        print(f"Epoch {ep_idx + 1}:")
        for s in sentences:
            print(f"  {s}")
    
    print("\n" + "="*60)
    print("ANALYSIS: Top-k Sampling vs Regular Sampling")
    print("="*60)
    print("""
Top-k sampling (k=5) typically produces:
1. MORE COHERENT text - by restricting sampling to only the top 5 most likely 
   characters, we avoid selecting unlikely characters that would break the flow.
   
2. LESS DIVERSE text - the restriction limits creativity and exploration.

3. MORE READABLE output - especially important when the vocabulary is large,
   as the probability mass spread across many unlikely tokens is eliminated.

The quality improvement comes from the fact that even very small probabilities
for many unlikely characters can add up, causing occasional "noise" characters
in regular sampling. Top-k eliminates this noise.
""")
    
    return model, data_handler


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    
    model, data_handler = train_model(
        train_path='train_shakespeare.txt',
        test_path='test_shakespeare.txt',
        block_size=10,
        n_layer=3,
        n_head=3,
        n_embd=48,
        learning_rate=3e-4,
        batch_size=64,
        epochs=10
    )
    

