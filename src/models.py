import torch
from transformers import BertModel


class MyModel(torch.nn.Module):
    def __init__(self, hidden_dim, ner_output_dim, dropout_prob=0.25):
        super().__init__()

        self.embedder = BertModel.from_pretrained('bert-base-uncased', ignore_mismatched_sizes=True)

        self.shared_lstm = torch.nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # # MODELO INICIAL
        # self.sa_fc = torch.nn.Linear(hidden_dim * 2, 1)  
        # self.ner_fc = torch.nn.Linear(hidden_dim * 2, ner_output_dim)  # will broadcast

        # INTENTANDO MEJORAR EL MODELO
        # Dropout for LSTM outputs and final layers
        self.dropout = torch.nn.Dropout(dropout_prob)

        # Layer normalization after LSTM
        self.ln_lstm = torch.nn.LayerNorm(hidden_dim * 2)

        # Sentiment Analysis head
        self.sa_fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(hidden_dim, 1)
        )

        # Named Entity Recognition head
        self.ner_fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(hidden_dim, ner_output_dim)
        )
    def forward(self, inputs):

        if inputs is None:
            raise ValueError("Input tensor cannot be None")
        
        if isinstance(inputs, torch.Tensor):
            embedder_output = self.embedder(inputs)
        else:
            raise ValueError("Inputs must be a Tensor")
    
        with torch.no_grad():
            embedder_output = self.embedder(inputs)
        
        embeddings = embedder_output.last_hidden_state


        lstm_out, (h_n, _) = self.shared_lstm(embeddings)
        lstm_out = self.ln_lstm(self.dropout(lstm_out))  # Intentando mejorar modelo

        # --- Sentiment Analysis ---
        # Take the final hidden state from both directions
        h_forward = h_n[-2]  # forward
        h_backward = h_n[-1]  # backward
        final_hidden = torch.cat((h_forward, h_backward), dim=1)
        final_hidden = self.dropout(final_hidden)  # Intentando mejorar modelo
        sa_output = self.sa_fc(final_hidden)  # [batch_size, sa_output_dim]

        # --- Named Entity Recognition ---
        ner_output = self.ner_fc(lstm_out)  # [batch_size, seq_len, ner_output_dim]

        return ner_output, sa_output
    

