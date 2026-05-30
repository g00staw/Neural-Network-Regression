import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from model import PriceRegressionModel
from load_data import X_train_processed, y_train, X_test_processed, y_test

torch.manual_seed(42)

X_train_t = torch.from_numpy(X_train_processed.astype(np.float32))
y_train_t  = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1)
X_test_t   = torch.from_numpy(X_test_processed.astype(np.float32))
y_test_t   = torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1)

BATCH_SIZE  = 256
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

input_size = X_train_t.shape[1]
model = PriceRegressionModel(input_size=input_size)

loss_fn   = nn.HuberLoss(delta=1.0)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10,
    min_lr=1e-6
)

best_val_loss    = float('inf')
best_model_state = None
patience         = 50
patience_counter = 0

EPOCHS = 500

for epoch in range(EPOCHS):

    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch in train_loader:
        y_pred     = model(X_batch)
        batch_loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()

    avg_train_loss = epoch_loss / len(train_loader)

    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_test_t)
        val_loss   = loss_fn(y_val_pred, y_test_t)

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping w epoce {epoch} | najlepsza val_loss: {best_val_loss:.4f}")
            break

    if epoch % 50 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoka {epoch:4d} | train_loss: {avg_train_loss:.4f} | val_loss: {val_loss.item():.4f} | lr: {current_lr:.6f}")

model.load_state_dict(best_model_state)

model.eval()
with torch.no_grad():
    y_final_pred = model(X_test_t).numpy()

y_pred_prices  = np.expm1(y_final_pred)
y_true_prices  = np.expm1(y_test_t.numpy())

mae  = np.mean(np.abs(y_pred_prices - y_true_prices))
rmse = np.sqrt(np.mean((y_pred_prices - y_true_prices) ** 2))


ss_res = np.sum((y_true_prices - y_pred_prices) ** 2)
ss_tot = np.sum((y_true_prices - np.mean(y_true_prices)) ** 2)
r2 = 1 - (ss_res / ss_tot)

mape = np.mean(np.abs((y_true_prices - y_pred_prices) / y_true_prices)) * 100

avg_price = np.mean(y_true_prices)

print(f"\n=== Wyniki końcowe ===")
print(f"Średnia cena w zbiorze testowym:  ${avg_price:.2f}")
print(f"─────────────────────────────────────────")
print(f"MAE   (średni błąd bezwzgl.):     ${mae:.2f}  ({mae/avg_price*100:.1f}% średniej ceny)")
print(f"RMSE  (pierwiastek błędu kwadr.): ${rmse:.2f}")
print(f"MAPE  (średni błąd %):             {mape:.1f}%  ← 'myli się średnio o X%'")
print(f"R²    (wyjaśniona wariancja):      {r2:.4f}  ({r2*100:.1f}% wariancji cen wyjaśnione)")
