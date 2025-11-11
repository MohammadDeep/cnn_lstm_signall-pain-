# cnn_lstm_signall-pain-


مدل‌های پایه‌ی **CNN**، **LSTM** و **CNN-LSTM** برای طبقه‌بندی سری‌های زمانی (به‌ویژه سیگنال‌های فیزیولوژیک مثل ECG/EEG/EDA/PPG/Respiration). تمرکز ریپو روی تعریف معماری‌ها و استفاده‌ی ساده از آن‌ها در PyTorch است؛ می‌توانید این مدل‌ها را در هر پایپ‌لاین آموزشیِ دلخواه ادغام کنید.

> ساختار ریپو شامل پوشه‌ی `base_modeles/` با فایل‌های مدل است. ([GitHub][1])

## فهرست

* [ویژگی‌ها](#ویژگیها)
* [پیش‌نیازها](#پیشنیازها)
* [ساختار پوشه‌ها](#ساختار-پوشهها)
* [نحوه‌ی استفاده سریع](#نحوهی-استفاده-سریع)
* [شکل ورودی‌ها](#شکل-ورودیها)
* [آموزش نمونه](#آموزش-نمونه)
* [ارزیابی و استنتاج](#ارزیابی-و-استنتاج)
* [نکات متفرقه](#نکات-متفرقه)
* [نقشه راه](#نقشه-راه)
* [لایسنس](#لایسنس)

## ویژگی‌ها

* معماری‌های ماژولار: **CNN1D**، **LSTM** و **CNN-LSTM** آماده‌ی استفاده.
* پشتیبانی از **Bidirectional LSTM**، **dropout** و روش‌های خلاصه‌سازی زمانی (مثل `last` یا `mean`) برای سناریوهای many-to-one.
* سازگار با داده‌های طول متغیر (با `pack_padded_sequence`) در صورت نیاز.
* طراحی ساده برای پلاگ‌کردن در هر لوپ آموزشی.

## پیش‌نیازها

* Python ≥ 3.9
* PyTorch ≥ 2.0
* numpy
* (اختیاری) scikit-learn برای معیارها

نصب سریع:

```bash
pip install torch numpy scikit-learn
```

## ساختار پوشه‌ها

```
.
├── base_modeles/
│   ├── cnn_model.py          # مدل CNN1D برای سری زمانی
│   ├── lstm_model.py         # مدل LSTM (یک‌طرفه/دوطرفه)
│   └── cnn_lstm_model.py     # معماری ترکیبی CNN + LSTM
└── README.md
```

> توجه: نام پوشه در حال حاضر `base_modeles` است (اگر می‌خواهید به `base_models` تغییر دهید از `git mv` استفاده کنید).

## نحوه‌ی استفاده سریع

### 1) وارد کردن و ساخت مدل

```python
import torch
from base_modeles.cnn_lstm_model import CNNLSTM1D    # اگر موجود است

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


# تست میانگین‌گیری زمانی
model2 = CNNLSTM1D(
    in_ch=C_in, n_classes=num_cls,
    cnn_channels=(16,), kernel_sizes=(7,),
    lstm_hidden=32, lstm_layers=2,   # dropout بین لایه‌ها فعال می‌شود
    bidirectional=False, dropout=0.2, aggregation='mean'
)
out2 = model2(x)
    

```

### 2) داده‌ی ساختگی برای تست سریع

```python
B, T, F = 8, 200, 16      # batch, طول توالی, تعداد ویژگی/کانال
x = torch.randn(B, T, F)  # برای LSTM: (B, T, F)
y = torch.randint(0, num_classes, (B,))

logits = model(x)         # (B, num_classes)
print(logits.shape)
```

## شکل ورودی‌ها

* **LSTM**: تنسور ورودی به‌شکل `(B, T, F)` وقتی `batch_first=True`

  * خروجی `output`: `(B, T, D*H)` و `(h_n, c_n)` به‌ترتیب `(D*L, B, H)`
  * `D=2` برای دوطرفه، `H=hidden_size`.
* **CNN1D**: معمولاً ورودی به‌شکل `(B, C, T)` است؛ اگر داده را به‌شکل `(B, T, F)` دارید، یکی را به‌عنوان `C` و دیگری را به‌عنوان `T` در نظر بگیرید (مثلاً `C=F` و `T=seq_len`) و قبل از عبور از CNN، محور‌ها را جابه‌جا کنید: `x = x.transpose(1, 2)` → `(B, F, T)`.
* **CNN-LSTM**: ابتدا چند لایه‌ی Conv1D روی `(B, C, T)` اعمال می‌شود، سپس خروجی به‌شکل `(B, T’, F’)` برای LSTM بازآرایی می‌گردد.

## آموزش نمونه

```python
import torch.nn as nn
import torch.optim as optim

model = LSTMClassifier(input_size=F, hidden_size=128, n_classes=num_classes,
                       bidirectional=True, aggregation='last', batch_first=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    logits = model(x)                 # x: (B, T, F)
    loss = criterion(logits, y)       # y: (B,)
    loss.backward()
    optimizer.step()
    print(f"epoch {epoch}: loss={loss.item():.4f}")
```

### طول‌های متغیر (اختیاری)

اگر توالی‌ها طول متفاوت دارند، از `pack_padded_sequence` استفاده کنید تا پَدینگ‌ها نادیده گرفته شود. سپس برای خلاصه‌سازی `last` از `h_n` یا اندیس‌گذاری با `lengths` استفاده کنید.

## ارزیابی و استنتاج

```python
model.eval()
with torch.no_grad():
    probs = model(x).softmax(dim=1)     # (B, num_classes)
    pred = probs.argmax(dim=1)          # (B,)
```

## نکات متفرقه

* در BiLSTM، برای نماینده‌ی «آخر توالی»، از `h_n` (یا الحاق `h_n[-2]` و `h_n[-1]`) استفاده کنید؛ نه `output[:, -1, :]`.
* اگر از CNN شروع می‌کنید و بعد LSTM، حتماً محور‌ها را بعد از Conv برگردانید تا شکل ورودی LSTM تبدیل به `(B, T', F')` شود.
* برای جلوگیری از overfitting از `dropout`، **early stopping** و **weight decay** استفاده کنید.

