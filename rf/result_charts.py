import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def feature_importance(best_model, feature_names, output_prefix=""):
    """Wykres ważności cech modelu Random Forest."""
    importances = best_model.feature_importances_

    indices = np.argsort(importances)[::-1]
    sorted_names = np.array(feature_names)[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=sorted_names,
                palette='viridis', hue=sorted_names, legend=False)

    plt.title('Random Forest — Ważność cech', fontsize=14)
    plt.xlabel('Ważność (Feature Importance)', fontsize=12)
    plt.ylabel('Cecha', fontsize=12)
    plt.tight_layout()

    file_name = f"{output_prefix}feature_importance.png"
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    print(f'Chart saved: {file_name}')


def pred_vs_actual(y_true, y_pred, output_prefix=""):
    """Wykres predykcja vs rzeczywistość — idealny model = punkty na linii y=x."""
    plt.figure(figsize=(8, 8))

    plt.scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue', label='Predykcje')

    # Linia idealna y=x
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Idealna predykcja')

    plt.xlabel('Prawdziwa cena ($)', fontsize=12)
    plt.ylabel('Przewidziana cena ($)', fontsize=12)
    plt.title('Predykcja vs Rzeczywistość', fontsize=14)
    plt.legend()
    plt.tight_layout()

    file_name = f"{output_prefix}pred_vs_actual.png"
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    print(f'Chart saved: {file_name}')


def residuals_plot(y_true, y_pred, output_prefix=""):
    """Wykres reszt (residuals) — reszty losowe wokół 0 = dobry model."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Lewy: reszty vs przewidziane wartości
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color='steelblue')
    axes[0].axhline(0, color='red', linestyle='--', lw=2)
    axes[0].set_xlabel('Przewidziana cena ($)', fontsize=12)
    axes[0].set_ylabel('Reszta (true - pred) ($)', fontsize=12)
    axes[0].set_title('Reszty vs Predykcja', fontsize=14)

    # Prawy: rozkład reszt
    axes[1].hist(residuals, bins=50, color='steelblue', edgecolor='white')
    axes[1].axvline(0, color='red', linestyle='--', lw=2)
    axes[1].set_xlabel('Reszta ($)', fontsize=12)
    axes[1].set_ylabel('Liczba próbek', fontsize=12)
    axes[1].set_title('Rozkład reszt', fontsize=14)

    plt.tight_layout()

    file_name = f"{output_prefix}residuals.png"
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    print(f'Chart saved: {file_name}')
