# ============================================================
# utils.py — Funzioni di supporto per logging, salvataggio e plot
# ============================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURAZIONE OUTPUT
# ============================================================

RESULTS_DIR = Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"
MODELS_DIR  = RESULTS_DIR / "models"
LOGS_DIR    = RESULTS_DIR / "logs"

def setup_dirs():
    """Crea le cartelle di output se non esistono."""
    for d in [RESULTS_DIR, PLOTS_DIR, MODELS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Cartelle di output pronte in: {RESULTS_DIR.resolve()}")

# ============================================================
# LOGGING
# ============================================================

def save_history(history, model_name):
    """Salva history di training come JSON."""
    setup_dirs()
    path = LOGS_DIR / f"{model_name}_history.json"
    with open(path, "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"History salvata: {path}")

def save_results_summary(results: dict, filename="results_summary.json"):
    """
    Salva un dizionario di risultati in JSON.
    Esempio: {"model": "CNN_v1", "val_accuracy": 0.62, "test_accuracy": 0.60}
    """
    setup_dirs()
    path = LOGS_DIR / filename
    # Carica risultati esistenti e aggiunge
    if path.exists():
        with open(path, "r") as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            existing = [existing]
    else:
        existing = []
    existing.append({**results, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Risultati salvati: {path}")

# ============================================================
# PLOT TRAINING CURVES
# ============================================================

def plot_history(history, title="Model", save=True):
    """
    Plot accuracy e loss su train e val.
    Salva il plot in results/plots/.
    """
    setup_dirs()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # Accuracy
    ax1.plot(history.history['accuracy'],     label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val',   linewidth=2, linestyle='--')
    ax1.set_title(f'{title} — Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    best_val = max(history.history['val_accuracy'])
    ax1.axhline(best_val, color='orange', linestyle=':', alpha=0.7,
                label=f'Best val: {best_val:.3f}')
    ax1.legend()

    # Loss
    ax2.plot(history.history['loss'],     label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val',   linewidth=2, linestyle='--')
    ax2.set_title(f'{title} — Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save:
        filename = title.replace(" ", "_").replace("/", "-") + ".png"
        path = PLOTS_DIR / filename
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Plot salvato: {path}")

    plt.show()
    print(f"  Miglior val_accuracy : {best_val:.4f}")
    print(f"  Ultima train_accuracy: {history.history['accuracy'][-1]:.4f}")

# ============================================================
# CONFRONTO TRA MODELLI
# ============================================================

def plot_comparison(histories: dict, metric="val_accuracy", save=True):
    """
    Confronta più modelli su una stessa metrica.
    histories = {"CNN v1": history1, "CNN v2": history2, ...}
    """
    setup_dirs()
    plt.figure(figsize=(10, 5))
    for name, hist in histories.items():
        plt.plot(hist.history[metric], label=name, linewidth=2)
    plt.title(f"Confronto modelli — {metric}")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = PLOTS_DIR / f"comparison_{metric}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Plot confronto salvato: {path}")

    plt.show()

# ============================================================
# VISUALIZZA PREDIZIONI
# ============================================================

def plot_predictions(model, generator, label_to_name, n=10, save=True, title="Predictions"):
    """
    Mostra n immagini con label reale e predetta.
    Verde = corretta, Rosso = errata.
    """
    setup_dirs()
    images, labels = next(iter(generator))
    preds = model.predict(images[:n], verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    plt.figure(figsize=(15, 4))
    for i in range(n):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i])
        true_name = label_to_name.get(str(int(labels[i])),  str(int(labels[i])))
        pred_name = label_to_name.get(str(int(pred_labels[i])), str(int(pred_labels[i])))
        correct   = int(labels[i]) == int(pred_labels[i])
        color     = "green" if correct else "red"
        plt.title(f"T: {true_name}\nP: {pred_name}", fontsize=6, color=color)
        plt.axis('off')
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()

    if save:
        path = PLOTS_DIR / f"{title.replace(' ', '_')}_predictions.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Plot predizioni salvato: {path}")

    plt.show()

# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(model, generator, label_to_name, save=True, title="Confusion Matrix"):
    """Calcola e plotta la confusion matrix (top classi per leggibilità)."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    setup_dirs()

    all_labels = []
    all_preds  = []

    for images, labels in generator:
        preds = model.predict(images, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1))
        all_labels.extend(labels.astype(int))

    cm = confusion_matrix(all_labels, all_preds)
    num_classes = cm.shape[0]

    # Nomi classi in ordine
    names = [label_to_name.get(str(i), str(i)) for i in range(num_classes)]

    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=names, yticklabels=names)
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(rotation=0,  fontsize=5)
    plt.tight_layout()

    if save:
        path = PLOTS_DIR / f"{title.replace(' ', '_')}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix salvata: {path}")

    plt.show()

# ============================================================
# SALVA / CARICA MODELLO
# ============================================================

def save_model(model, model_name):
    """Salva i pesi del modello."""
    setup_dirs()
    path = MODELS_DIR / f"{model_name}.keras"
    model.save(path)
    print(f"Modello salvato: {path}")

def load_model(model_name):
    """Carica un modello salvato."""
    import tensorflow as tf
    path = MODELS_DIR / f"{model_name}.keras"
    model = tf.keras.models.load_model(path)
    print(f"Modello caricato: {path}")
    return model