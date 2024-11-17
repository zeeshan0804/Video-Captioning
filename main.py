import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import AdamW
import os
from utils import preprocess_data, create_caption_pipeline
from model import VideoCaptioningModel

# Load and preprocess data
file_path = 'data/Topic Modeling/test.csv'
df = preprocess_data(file_path)
input_ids, attention_masks, label_ids = create_caption_pipeline(df)

# Create DataLoader
dataset = TensorDataset(input_ids, attention_masks, label_ids)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

print(f'\nTotal dataset size: {len(dataset)}')
print(f'Training set size: {train_size}')
print(f'Validation set size: {val_size}\n')

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize model and training
model = VideoCaptioningModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # Added weight_decay parameter
criterion = torch.nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)

# Early stopping parameters
patience = 3
best_val_loss = float('inf')
patience_counter = 0
model_save_dir = 'checkpoints'
os.makedirs(model_save_dir, exist_ok=True)

# Training loop
epochs = 5
for epoch in range(epochs):
    # Training
    model.train()
    total_loss = 0
    for batch in train_loader:
        b_input_ids, b_attention_mask, b_labels = [x.to(device) for x in batch]
        
        optimizer.zero_grad()
        outputs = model(b_input_ids, b_attention_mask, b_labels)
        loss = criterion(outputs.view(-1, outputs.size(-1)), b_labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_train_loss = total_loss/len(train_loader)
    print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss}')
    
    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            b_input_ids, b_attention_mask, b_labels = [x.to(device) for x in batch]
            outputs = model(b_input_ids, b_attention_mask, b_labels)
            loss = criterion(outputs.view(-1, outputs.size(-1)), b_labels.view(-1))
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss/len(val_loader)
    print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss}')
    
    # Save model checkpoint for each epoch
    checkpoint_path = os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, checkpoint_path)
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, os.path.join(model_save_dir, 'best_model.pth'))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping triggered after epoch {epoch+1}')
            break

# Load best model for inference
best_model_path = os.path.join(model_save_dir, 'best_model.pth')
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Example inference
model.eval()
with torch.no_grad():
    sample_input_ids = input_ids[:1].to(device)
    sample_attention_mask = attention_masks[:1].to(device)
    caption = model.generate_caption(sample_input_ids, sample_attention_mask)[0]
    print(f'Generated Caption: {caption}')
