import torch
import torch.nn as nn
from transformers import RobertaModel, logging

logging.set_verbosity_error()

class RobertaBiLSTM(nn.Module):
    def __init__(self, num_classes=7, hidden_dim=256, dropout=0.3, pooling_mode='mean'):
        super(RobertaBiLSTM, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.lstm = nn.LSTM(768, hidden_dim, batch_first=True, bidirectional=True)
        self.pooling_mode = pooling_mode
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # lstm_out shape: [Batch, Seq_Len, hidden_dim * 2]
        lstm_out, (h_n, _) = self.lstm(outputs.last_hidden_state)

        # Apply attention mask to ignore padding tokens in pooling
        mask = attention_mask.unsqueeze(-1).expand(lstm_out.size()).float()
        masked_out = lstm_out * mask

        if self.pooling_mode == 'max':
            # Replace padding with a very small number so it's not picked as 'max'
            # We use -1e9 for the masked areas
            neg_inf_mask = (1 - mask) * -1e9
            combined, _ = torch.max(masked_out + neg_inf_mask, dim=1)
            
        elif self.pooling_mode == 'last':
            # Concatenate the final forward and backward hidden states
            combined = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
            
        else: # 'mean'
            sum_embeddings = torch.sum(masked_out, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            combined = sum_embeddings / sum_mask

        return self.classifier(self.dropout(combined))