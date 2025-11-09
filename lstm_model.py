import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMClassifier(nn.Module):
    """
    LSTM برای طبقه‌بندی توالی‌ها.
    ورودی: x با شکل (B, T, F) اگر batch_first=True (پیش‌فرض)
    lengths: طول واقعی هر نمونه (اختیاری؛ برای توالی‌های با پدینگ)

    aggregation:
      - 'last' : استفاده از آخرین حالت مخفی (ترجیحی و دقیق‌تر)
      - 'mean' : میانگین‌گیری روی زمان (با ماسک طول‌ها)
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_classes: int,
                 num_layers: int = 1,
                 bidirectional: bool = True,
                 dropout: float = 0.0,        # فقط وقتی num_layers>1 اعمال می‌شود
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
        feat_dim = self.num_dirs * hidden_size
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        """
        x: (B, T, F) اگر batch_first=True
        lengths: (B,) طول واقعی هر سکوئنس (CPU یا همان device؛ اگر pack می‌کنی بهتر است روی CPU باشد)
        """
        if lengths is not None:
            # برای صرفه‌جویی محاسباتی روی پدینگ
            # اطمینان از CPU بودن lengths برای pack (نسخه‌های جدید GPU هم اوکی‌اند، ولی CPU مطمئن‌تر است)
            if lengths.device.type != 'cpu':
                lengths_cpu = lengths.to('cpu')
            else:
                lengths_cpu = lengths

            packed = pack_padded_sequence(x, lengths_cpu, batch_first=self.batch_first, enforce_sorted=False)
            y_packed, (hn, cn) = self.lstm(packed)

            if self.aggregation == 'last':
                # hn: (num_layers * num_dirs, B, hidden_size)
                nl = self.lstm.num_layers
                nd = self.num_dirs
                hn = hn.view(nl, nd, x.size(0), self.hidden_size)  # (nl, nd, B, H)
                h_last = hn[-1].transpose(0, 1).reshape(x.size(0), nd * self.hidden_size)  # (B, nd*H)
                feats = h_last
            elif self.aggregation == 'mean':
                y, _ = pad_packed_sequence(y_packed, batch_first=self.batch_first)  # (B, T_max, nd*H)
                B, T_max, D = y.shape
                # ماسکِ طول‌ها
                device = y.device
                rng = torch.arange(T_max, device=device).unsqueeze(0)  # (1, T_max)
                mask = (rng < lengths.unsqueeze(1).to(device)).float()  # (B, T_max)
                feats = (y * mask.unsqueeze(-1)).sum(dim=1) / lengths.clamp_min(1).unsqueeze(1).to(device)  # (B, D)
            else:
                raise ValueError("aggregation باید 'last' یا 'mean' باشد.")
        else:
            # بدون lengths: ساده
            y, (hn, cn) = self.lstm(x)  # y: (B, T, nd*H) اگر batch_first=True
            if self.aggregation == 'last':
                # راه ۱: از hn (ایمن‌تر)
                nl = self.lstm.num_layers
                nd = self.num_dirs
                hn = hn.view(nl, nd, x.size(0), self.hidden_size)
                feats = hn[-1].transpose(0, 1).reshape(x.size(0), nd * self.hidden_size)
                # راه ۲ (جایگزین): feats = y[:, -1, :]
            elif self.aggregation == 'mean':
                feats = y.mean(dim=1)
            else:
                raise ValueError("aggregation باید 'last' یا 'mean' باشد.")

        logits = self.head(feats)  # (B, n_classes)
        return logits
