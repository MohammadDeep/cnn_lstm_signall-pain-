import torch
import torch.nn as nn

class LSTM_model(nn.Module):
    """
    LSTM برای طبقه‌بندی توالی‌های هم‌طول.
    ورودی: x با شکل (B, T, F)  (batch_first=True)
    aggregation: 'last' یا 'mean'
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_classes: int,
                 num_layers: int = 1,
                 bidirectional: bool = True,
                 dropout: float = 0.0,     # فقط وقتی num_layers>1 اعمال می‌شود
                 aggregation: str = 'last',
                 batch_first: bool = True):
        super().__init__()
        self.batch_first = batch_first
        self.aggregation = aggregation
        self.num_dirs = 2 if bidirectional else 1
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=(dropout if num_layers > 1 else 0.0),
            batch_first=batch_first,
        )
        self.head = nn.Linear(self.num_dirs * hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # برای کارایی بهتر روی cuDNN (به‌خصوص GPU)
        self.lstm.flatten_parameters()

        # y: (B, T, nd*H) اگر batch_first=True
        y, (hn, cn) = self.lstm(x)

        if self.aggregation == 'last':
            # hn: (num_layers * num_dirs, B, H)
            nl = self.lstm.num_layers
            nd = self.num_dirs
            B  = x.size(0)
            h_last = hn.view(nl, nd, B, self.hidden_size)[-1]   # (nd, B, H)
            feats  = h_last.transpose(0, 1).reshape(B, nd * self.hidden_size)  # (B, nd*H)
        elif self.aggregation == 'mean':
            feats = y.mean(dim=1)  # میانگین روی زمان (T)
        else:
            raise ValueError("aggregation باید 'last' یا 'mean' باشد.")

        logits = self.head(feats)  # (B, n_classes)
        return logits
