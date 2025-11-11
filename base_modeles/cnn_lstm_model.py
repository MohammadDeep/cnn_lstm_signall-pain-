# نکته یادت باشد باید ابعاد خروجی کانوشن را تغییر شکل دهی .
import torch
import torch.nn as nn

import torch
try:
    from cnn_model import ConvBNAct1d           # اگر موجود است
    from lstm_model import LSTM_model           # اگر موجود است
except:
    from base_modeles.cnn_model import ConvBNAct1d           # اگر موجود است
    from base_modeles.lstm_model import LSTM_model   


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
        self.lstm_head = LSTM_model(
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
