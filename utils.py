import pandas as pd
import torch
from transformers import BertTokenizer

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # Handle missing values
    df.dropna(subset=['ABSTRACT', 'TITLE'], inplace=True)
    return df

def create_caption_pipeline(df, max_length=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def tokenize_function(text, target):
        inputs = tokenizer(text, padding='max_length', truncation=True, 
                         max_length=max_length, return_tensors='pt')
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(target, padding='max_length', truncation=True, 
                             max_length=max_length, return_tensors='pt')
        return inputs, labels
    
    encoded_pairs = [tokenize_function(transcript, caption) 
                    for transcript, caption in zip(df['transcript'], df['caption'])]
    
    input_ids = torch.cat([x[0]['input_ids'] for x in encoded_pairs])
    attention_masks = torch.cat([x[0]['attention_mask'] for x in encoded_pairs])
    label_ids = torch.cat([x[1]['input_ids'] for x in encoded_pairs])
    
    return input_ids, attention_masks, label_ids

# Example usage:
# df = preprocess_data('/Users/zeeshan/Video Analyser/Video-Captioning/data/Topic Modeling/train.csv')
# input_ids, attention_masks, labels = create_caption_pipeline(df, max_length=128)
