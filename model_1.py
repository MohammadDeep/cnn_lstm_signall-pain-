import torch
from base_modeles.cnn_lstm_model import CNNLSTM1D    # اگر موجود است
 

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
