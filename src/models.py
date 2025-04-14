import torch


class MyModel(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, sa_output_dim, ner_output_dim):
        super().__init__()
        self.shared_lstm = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.sa_fc = torch.nn.Linear(hidden_dim * 2, sa_output_dim)  
        self.ner_fc = torch.nn.Linear(hidden_dim * 2, ner_output_dim)  # will broadcast

    def forward(self, x):
        lstm_out, (h_n, _) = self.shared_lstm(x)

        # --- Sentiment Analysis ---
        # Take the final hidden state from both directions
        h_forward = h_n[-2]  # forward
        h_backward = h_n[-1]  # backward
        final_hidden = torch.cat((h_forward, h_backward), dim=1)
        sa_output = self.sa_fc(final_hidden)  # [batch_size, sa_output_dim]

        # --- Named Entity Recognition ---
        ner_output = self.ner_fc(lstm_out)  # [batch_size, seq_len, ner_output_dim]

        return sa_output, ner_output