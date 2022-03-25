from torch import nn, sigmoid


class Classification(nn.Module):
    def __init__(self, d_model, output_dim=2):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # [batch, d_model]
        x = sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x


if __name__ == '__main__':
    import torch
    batch_size = 5
    tensor = torch.rand([batch_size, 768])
    classifier = Classification(d_model=768, output_dim=2)
    print(classifier(tensor))
