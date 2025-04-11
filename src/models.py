import torch


class NerSAModel(torch.nn.Module):
    
    def __init__(self, hidden_size, num_layers):
        """
        This method is the constructor of the class.

        Args:
            hidden_size: hidden size of the RNN layers
            num_layers: number of RNN layers
        """

        # TODO
        super().__init__()
        input_size = 24
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
        )
        self.fc = torch.nn.Linear(hidden_size, 24)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: inputs tensor. Dimensions: [batch, number of past days, 24].

        Returns:
            output tensor. Dimensions: [batch, 24].
        """

        # TODO
        batch_size = inputs.shape[0]
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=inputs.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=inputs.device)

        outputs, _ = self.lstm(inputs, (h0, c0))
        final_hidden_state = outputs[:, -1, :]

        return self.fc(final_hidden_state)


# input -> red -> output intermedio (embedding/representacion)
# -> dos modelos distintos (sa + ner) 
    
# crear data.py con dataset
# train_functions (train_step, etc)
