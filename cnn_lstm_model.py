# نکته یادت باشد باید ابعاد خروجی کانوشن را تغییر شکل دهی .
import torch
import torch.nn as nn

# === از کد خودت: ConvBNAct1d ===
class ConvBNAct1d(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, s=1, d=1, groups=1, act='relu'):
        super().__init__()
        p = (k // 2) * d  # same-ish برای S=1
        self.conv = nn.Conv1d(in_ch, out_ch, k, stride=s, padding=p,
                              dilation=d, groups=groups, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.ReLU(inplace=True) if act == 'relu' else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# === از کد خودت: LSTMClassifier (نسخهٔ هم‌طول) ===
class LSTMClassifier(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_classes: int,
                 num_layers: int = 1,
                 bidirectional: bool = True,
                 dropout: float = 0.0,
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
        self.lstm.flatten_parameters()
        y, (hn, cn) = self.lstm(x)
        if self.aggregation == 'last':
            nl = self.lstm.num_layers
            nd = self.num_dirs
            B  = x.size(0)
            h_last = hn.view(nl, nd, B, self.hidden_size)[-1]   # (nd, B, H)
            feats  = h_last.transpose(0, 1).reshape(B, nd * self.hidden_size)
        elif self.aggregation == 'mean':
            feats = y.mean(dim=1)
        else:
            raise ValueError("aggregation باید 'last' یا 'mean' باشد.")
        logits = self.head(feats)
        return logits


# === مدل ترکیبی: CNN → LSTM ===
class CNNLSTM1D(nn.Module):
    """
    ورودی:  x با شکل (B, C_in, T)
    جریان:  ConvBNAct1d × N  →  (B, C_feat, T)  → permute به (B, T, C_feat)  →  LSTMClassifier
    خروجی:  logits با شکل (B, n_classes)
    """
    def __init__(self,
                 in_ch: int,
                 n_classes: int,
                 cnn_channels=(32, 64),      # می‌تونی لایه‌ها را کم/زیاد کنی
                 kernel_sizes=(7, 5),
                 lstm_hidden: int = 128,
                 lstm_layers: int = 1,
                 bidirectional: bool = True,
                 dropout: float = 0.3,
                 aggregation: str = 'last'):
        super().__init__()
        assert len(cnn_channels) == len(kernel_sizes), "طول cnn_channels و kernel_sizes باید برابر باشد"

        # CNN feature extractor (حفظ طول زمانی با padding same-ish)
        blocks = []
        c_in = in_ch
        for c_out, k in zip(cnn_channels, kernel_sizes):
            blocks.append(ConvBNAct1d(c_in, c_out, k=k))
            c_in = c_out
        self.cnn = nn.Sequential(*blocks)

        # LSTM head: input_size = آخرین تعداد کانال‌های CNN
        feat_dim = cnn_channels[-1]
        self.lstm_head = LSTMClassifier(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            n_classes=n_classes,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            aggregation=aggregation,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        z = self.cnn(x)            # (B, C_feat, T)  با طول زمانی حفظ‌شده
        z = z.permute(0, 2, 1)     # (B, T, C_feat)  مطابق انتظار LSTM
        logits = self.lstm_head(z) # (B, n_classes)
        return logits


# ====== تست‌های کوتاه (smoke tests) ======
if __name__ == "__main__":
    B, C_in, T = 4, 4, 256
    num_cls = 6

    model = CNNLSTM1D(
        in_ch=C_in, n_classes=num_cls,
        cnn_channels=(32, 64), kernel_sizes=(7, 5),
        lstm_hidden=64, lstm_layers=1,
        bidirectional=True, dropout=0.3, aggregation='last'
    )

    x = torch.randn(B, C_in, T)
    out = model(x)
    print("logits shape:", out.shape)   # باید (B, num_cls) باشد

    # تست میانگین‌گیری زمانی
    model2 = CNNLSTM1D(
        in_ch=C_in, n_classes=num_cls,
        cnn_channels=(16,), kernel_sizes=(7,),
        lstm_hidden=32, lstm_layers=2,   # dropout بین لایه‌ها فعال می‌شود
        bidirectional=False, dropout=0.2, aggregation='mean'
    )
    out2 = model2(x)
    print("logits shape (mean):", out2.shape)
