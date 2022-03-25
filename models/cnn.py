from torch import nn, relu, sigmoid, reshape, flatten


class Cnn(nn.Module):
    def __init__(self, seq_len, d_model, output_dim):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 1, (2, 768))
        #  self.pool1 = nn.MaxPool2d(1)

        self.linear1 = nn.Linear(127, d_model)
        self.linear_last = nn.Linear(d_model, output_dim)

        #  self.drop = nn.Dropout()

    def forward(self, x):
        # [batch, 128, 768]
        x_shape = x.shape
        x = reshape(x, (x_shape[0], 1, *x_shape[1:]))
        x = relu(self.conv1(x))
        #  x = self.pool1(x)
        x = flatten(x, 1)
        x = sigmoid(self.linear1(x))
        x = self.linear_last(x)
        return x


if __name__ == '__main__':
    import torch
    batch_size = 5
    tensor = torch.rand([batch_size, 128, 768])
    classifier = Cnn(seq_len=128, d_model=768, output_dim=2)
    print(classifier(tensor))
