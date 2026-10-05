import matplotlib
matplotlib.use("Agg")  # no GUI needed (also avoids tkinter problems on Windows)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


def save_confusion_matrix(cm, class_names, save_path):
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, data, fmt, title in ((axes[0], cm, "d", "Confusion matrix (counts)"),
                                 (axes[1], norm, ".2f", "Confusion matrix (row-normalised)")):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names,
                    yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_curves(history, save_path, head_epochs=0):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (a, b, title) in zip(axes[:2], (("train_loss", "val_loss", "Loss"),
                                            ("train_acc", "val_acc", "Accuracy"))):
        ax.plot(ep, history[a], label="train")
        ax.plot(ep, history[b], label="validation")
        ax.set_title(title)
    axes[2].plot(ep, history["val_f1"], label="val macro-F1")
    axes[2].plot(ep, history["val_bin_acc"], label="val real-vs-AI accuracy")
    axes[2].set_title("Validation metrics")
    for ax in axes:
        if 0 < head_epochs < len(history["train_loss"]):
            ax.axvline(head_epochs + 0.5, color="grey", linestyle=":", label="unfreeze backbone")
        ax.set_xlabel("epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_roc_curve(y_true, y_probs, class_names, save_path):
    k = len(class_names)
    plt.figure(figsize=(8, 6))
    if k == 2:
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.4f}")
    else:
        y_bin = label_binarize(y_true, classes=list(range(k)))
        for i in range(k):
            if y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            plt.plot(fpr, tpr, label=f"{class_names[i]} AUC = {auc(fpr, tpr):.4f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_binary_roc(is_ai, ai_score, save_path):
    fpr, tpr, _ = roc_curve(is_ai, ai_score)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"real vs AI, AUC = {auc(fpr, tpr):.4f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False positive rate (real flagged as AI)")
    plt.ylabel("True positive rate (AI detected)")
    plt.title("Real vs AI-generated ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
