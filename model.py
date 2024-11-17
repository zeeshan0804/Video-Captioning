from transformers import BertTokenizer, EncoderDecoderModel
import torch
import torch.nn as nn

class VideoCaptioningModel(nn.Module):
    def __init__(self):
        super(VideoCaptioningModel, self).__init__()
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def forward(self, input_ids, attention_mask, decoder_input_ids=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )
        return outputs.logits
    
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
