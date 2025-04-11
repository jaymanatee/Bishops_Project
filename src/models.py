import torch


class MyModel(torch.nn.Module):
    
    def __init__(self, hidden_size, num_layers, representation_size):
        super().__init__()
        input_size = 24

        self.base_model = torch.nn.Sequential(
            torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                num_layers=num_layers,
                bidirectional=True
            ),
            torch.nn.Linear(hidden_size, representation_size)
        )
        self.sa = torch.nn.Sequential(
            torch.nn.Linear(representation_size, 2)
        )
        self.ner = torch.nn.Sequential(
            torch.nn.Linear(representation_size, input_size)
        )


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        representation = self.base_model(inputs)
        return self.ner(representation), self.sa(representation)


# input -> red -> output intermedio (embedding/representacion)
# -> dos modelos distintos (sa + ner) 
    
# crear data.py con dataset
# train_functions (train_step, etc)
