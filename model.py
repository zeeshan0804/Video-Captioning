from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import torch.nn as nn

class VideoCaptioningModel(nn.Module):
    def __init__(self):
        super(VideoCaptioningModel, self).__init__()
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        
        # Configure model parameters
        self.model.config.max_length = 128
        self.model.config.min_length = 10
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.early_stopping = True
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs.logits if labels is None else outputs
    
    def generate_caption(self, input_ids, attention_mask, max_length=64):
        caption_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        captions = self.tokenizer.batch_decode(caption_ids, skip_special_tokens=True)
        return captions

# Example usage:
# model = VideoCaptioningModel()
# input_ids, attention_masks = create_bert_pipeline(df, 'text_column_name')
# logits = model(input_ids, attention_masks)
# captions = model.generate_caption(input_ids, attention_masks)
