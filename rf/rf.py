import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
import os

# Dodaj katalog główny projektu do sys.path — potrzebne dla load_data.py
# (rf/ jest dodawane automatycznie przez Python przy uruchomieniu skryptu)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from load_data import X_train_processed, y_train, X_test_processed, y_test, preprocessor
from result_charts import feature_importance, pred_vs_actual, residuals_plot

print("Start GridSearch for best parameters...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 15, 25],
    'min_samples_split': [2, 5],
}

rf_base = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=rf_base,
                           param_grid=param_grid,
                           cv=3,
                           scoring='neg_mean_absolute_error',  # optymalizuj MAE
                           n_jobs=-1,
                           verbose=2)

grid_search.fit(X_train_processed, y_train)

print("\nBest parameters:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

# --- Predykcja i metryki ---
y_pred_log = best_model.predict(X_test_processed)

# Odwróć transformację log1p → oryginalne ceny w $
y_pred_prices = np.expm1(y_pred_log)
y_true_prices = np.expm1(y_test)

mae  = mean_absolute_error(y_true_prices, y_pred_prices)
rmse = np.sqrt(mean_squared_error(y_true_prices, y_pred_prices))
r2   = r2_score(y_true_prices, y_pred_prices)
mape = np.mean(np.abs((y_true_prices - y_pred_prices) / y_true_prices)) * 100
avg  = np.mean(y_true_prices)

print("\n=== Wyniki końcowe ===")
print(f"Średnia cena w zbiorze testowym:  ${avg:.2f}")
print(f"─────────────────────────────────────────")
print(f"MAE   (średni błąd bezwzgl.):     ${mae:.2f}  ({mae/avg*100:.1f}% średniej ceny)")
print(f"RMSE  (pierwiastek błędu kwadr.): ${rmse:.2f}")
print(f"MAPE  (średni błąd %):             {mape:.1f}%")
print(f"R²    (wyjaśniona wariancja):      {r2:.4f}  ({r2*100:.1f}%)")

# Nazwy cech po preprocessingu (ColumnTransformer zmienia nazwy kolumn)
feature_names = preprocessor.get_feature_names_out()

_charts_prefix = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts", "rf_")

# --- Wykresy ---
feature_importance(best_model, feature_names, output_prefix=_charts_prefix)
pred_vs_actual(y_true_prices, y_pred_prices, output_prefix=_charts_prefix)
residuals_plot(y_true_prices, y_pred_prices, output_prefix=_charts_prefix)
