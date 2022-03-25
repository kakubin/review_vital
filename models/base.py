from models.positional_encoder import PositionalEncoder
from models.classification import Classification
from torch import nn


class TransformerModel(nn.Module):
    """base model"""

    def __init__(self, seq_len=512, d_model=768):
        super().__init__()
        self.positional_encoder = PositionalEncoder(
                seq_len=seq_len,
                d_model=d_model
                )

        encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                batch_first=True,
                )
        self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=4,
                )

        self.classification = Classification(d_model, output_dim=2)

    def forward(self, x, padding_mask):
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = x[:, 0, :]
        x = self.classification(x)
        return x
