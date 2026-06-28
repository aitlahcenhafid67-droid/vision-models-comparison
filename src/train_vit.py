"""
ÉTAPE 3B : Fine-tuning de ViT (Vision Transformer)
====================================================
ViT est un modèle de CLASSIFICATION d'images.
Il dit ce qu'il voit dans l'image, sans localiser.

Comment ViT fonctionne :
- L'image est découpée en petits patchs (carrés 16x16 pixels)
- Ces patchs sont traités comme des "mots" dans un Transformer
- La dernière couche prédit la classe

Optimisations pour CPU (votre machine n'a pas de GPU) :
- On gèle l'encodeur ViT (300MB de paramètres)
- On fine-tune SEULEMENT la couche classifier (3×768 = 2304 params)
- Cette technique s'appelle "linear probing" ou "head-only fine-tuning"
- On utilise un sous-ensemble de 800 images pour aller plus vite
- On peut ensuite "dégeler" quelques couches si on veut aller plus loin

Sorties après entraînement :
- models/vit_finetuned/          (poids HuggingFace)
- models/vit_finetuned/metrics_vit.json
- models/vit_finetuned/training_curves_vit.png
"""

import sys
import time
import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend non-interactif pour Windows
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import classification_report

from src.utils.metrics import compute_classification_metrics, save_metrics


# ── Constantes ────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["Bird", "Cat", "Dog"]
MODEL_NAME   = "google/vit-base-patch16-224"
IMAGE_SIZE   = 224


def get_transforms(is_train: bool = True) -> transforms.Compose:
    """
    Transformations appliquées aux images.
    Pour l'entraînement : augmentations légères pour apprendre mieux.
    Pour l'évaluation   : juste redimensionner et normaliser.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])


def load_datasets(vit_dir: Path, max_train: int = 800, max_val: int = 200):
    """
    Charge les datasets avec une limite d'images pour accélérer sur CPU.

    Args:
        vit_dir  : Dossier racine des crops ViT
        max_train: Nombre max d'images d'entraînement (limité pour CPU)
        max_val  : Nombre max d'images de validation
    """
    train_dir = vit_dir / "train"
    val_dir   = vit_dir / "val"
    test_dir  = vit_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        if not d.exists():
            raise FileNotFoundError(
                f"Dossier manquant : {d}\n"
                "Lancez d'abord : python src/prepare_dataset.py"
            )

    full_train = datasets.ImageFolder(str(train_dir), transform=get_transforms(True))
    full_val   = datasets.ImageFolder(str(val_dir),   transform=get_transforms(False))
    full_test  = datasets.ImageFolder(str(test_dir),  transform=get_transforms(False))

    # Sous-échantillonnage stratifié (équilibré par classe) pour accélérer sur CPU
    train_ds = _balanced_subset(full_train, max_train)
    val_ds   = _balanced_subset(full_val,   max_val)
    test_ds  = full_test   # test set complet pour les métriques réelles

    print(f"  Train : {len(train_ds)} images (sur {len(full_train)} total)")
    print(f"  Val   : {len(val_ds)} images")
    print(f"  Test  : {len(test_ds)} images (complet)")
    print(f"  Classes : {full_train.classes}")

    return train_ds, val_ds, test_ds, full_train.classes


def _balanced_subset(dataset: datasets.ImageFolder, max_total: int):
    """
    Crée un sous-ensemble équilibré (même nombre d'images par classe).

    Exemple : max_total=800, 3 classes → 266 images par classe
    """
    if len(dataset) <= max_total:
        return dataset

    per_class = max_total // len(dataset.classes)
    indices = []

    for class_idx in range(len(dataset.classes)):
        class_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
        random.shuffle(class_indices)
        indices.extend(class_indices[:per_class])

    return Subset(dataset, indices)


def build_model_frozen(num_classes: int, device: torch.device):
    """
    Charge ViT et gèle tout sauf la couche classifier.

    Pourquoi geler ?
    - ViT a 86 millions de paramètres → très lent à entraîner sur CPU
    - Les couches profondes ont déjà appris à "voir" des formes et textures
    - On change juste la "tête" finale pour décider entre Bird/Cat/Dog

    Paramètres entraînables : seulement 3×768 + 3 = 2307 (au lieu de 86M!)
    """
    print(f"  Chargement de {MODEL_NAME}...")

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    # Geler TOUS les paramètres
    for param in model.parameters():
        param.requires_grad = False

    # Dégeler SEULEMENT la couche classifier (tête de classification)
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres total      : {total:,}")
    print(f"  Paramètres entraîn.   : {trainable:,} ({trainable/total:.2%} seulement)")

    return model.to(device)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Un seul passage d'entraînement.
    Retourne : (perte_moyenne, accuracy_moyenne)
    """
    model.train()
    total_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=images)
        loss    = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds       = outputs.logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Évalue le modèle. Retourne (perte, accuracy, vrais, prédits).
    """
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(pixel_values=images)
        loss    = criterion(outputs.logits, labels)

        total_loss += loss.item()
        preds       = outputs.logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / len(loader), correct / total, all_labels, all_preds


def save_training_curves(history: dict, save_path: Path) -> None:
    """Sauvegarde les courbes loss et accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train", markersize=4)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val",   markersize=4)
    axes[0].set_title("Perte (Loss) par époque")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Perte")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train", markersize=4)
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val",   markersize=4)
    axes[1].set_title("Accuracy par époque")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle("ViT Fine-tuning - Courbes d'Apprentissage", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Courbes sauvegardées : {save_path}", flush=True)


def train_vit(
    project_root: str,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,   # LR plus élevé car on n'entraîne que la tête
    max_train_samples: int = 800,
) -> dict:
    """
    Fine-tuning ViT optimisé pour CPU.

    Stratégie : head-only fine-tuning
    - Encodeur gelé → extraction de features rapide
    - Seule la couche finale est entraînée
    - Résultats corrects en ~10-20 min sur CPU
    """
    project_root = Path(project_root)
    vit_dir      = project_root / "data" / "vit_crops"
    save_dir     = project_root / "models" / "vit_finetuned"
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55, flush=True)
    print("  FINE-TUNING ViT (Vision Transformer)", flush=True)
    print("  Stratégie : head-only (encoder gelé)", flush=True)
    print("=" * 55, flush=True)
    print(f"  Device        : {device}", flush=True)
    print(f"  Époques       : {epochs}", flush=True)
    print(f"  Batch size    : {batch_size}", flush=True)
    print(f"  Images train  : max {max_train_samples}", flush=True)
    print(flush=True)

    # ── Données ──────────────────────────────────────────────────────────────
    print("[1/4] Chargement des données...", flush=True)
    train_ds, val_ds, test_ds, classes = load_datasets(vit_dir, max_train_samples)
    num_classes = len(classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # ── Modèle ───────────────────────────────────────────────────────────────
    print("\n[2/4] Chargement du modèle ViT...", flush=True)
    model = build_model_frozen(num_classes, device)

    # ── Entraînement ─────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    # LR plus élevé (1e-3) car on entraîne seulement la tête linéaire
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    start_time   = time.time()

    print(f"\n[3/4] Entraînement ({epochs} époques)...", flush=True)
    print(f"  Durée estimée : ~{epochs * 2}-{epochs * 5} min sur CPU", flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  Époque {epoch:2d}/{epochs} | "
            f"Loss: {train_loss:.4f} Acc: {train_acc:.1%} | "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.1%} | "
            f"{epoch_time:.0f}s",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(str(save_dir))
            print(f"           -> Sauvegarde (val_acc={val_acc:.1%})", flush=True)

    training_time = time.time() - start_time

    # ── Courbes ──────────────────────────────────────────────────────────────
    save_training_curves(history, save_dir / "training_curves_vit.png")

    # ── Test final ───────────────────────────────────────────────────────────
    print("\n[4/4] Évaluation finale...", flush=True)
    best_model = ViTForImageClassification.from_pretrained(str(save_dir)).to(device)
    _, test_acc, y_true, y_pred = evaluate(best_model, test_loader, criterion, device)

    clf = compute_classification_metrics(y_true, y_pred, CLASS_NAMES)
    print("\n  Rapport de classification :")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    model_size_mb = sum(
        f.stat().st_size for f in save_dir.rglob("*.bin") if f.is_file()
    ) / (1024 * 1024)
    # Pour les nouveaux transformers, les poids sont en safetensors
    if model_size_mb == 0:
        model_size_mb = sum(
            f.stat().st_size for f in save_dir.rglob("*.safetensors") if f.is_file()
        ) / (1024 * 1024)

    metrics = {
        "model"                : "ViT-Base-Patch16-224 (head-only)",
        "task"                 : "classification",
        "strategy"             : "frozen encoder + trained head",
        "precision"            : round(clf["precision"], 4),
        "recall"               : round(clf["recall"], 4),
        "f1"                   : round(clf["f1"], 4),
        "accuracy"             : round(test_acc, 4),
        "training_time_seconds": round(training_time, 1),
        "training_time_minutes": round(training_time / 60, 1),
        "model_size_mb"        : round(model_size_mb, 2),
        "epochs_trained"       : epochs,
        "best_val_acc"         : round(best_val_acc, 4),
        "train_samples"        : max_train_samples,
    }

    save_metrics(metrics, save_dir / "metrics_vit.json")
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 55, flush=True)
    print("  RÉSULTATS ViT", flush=True)
    print("=" * 55, flush=True)
    print(f"  Précision   : {clf['precision']:.1%}", flush=True)
    print(f"  Rappel      : {clf['recall']:.1%}", flush=True)
    print(f"  F1-Score    : {clf['f1']:.1%}", flush=True)
    print(f"  Accuracy    : {test_acc:.1%}", flush=True)
    print(f"  Taille      : {model_size_mb:.1f} MB", flush=True)
    print(f"  Durée       : {training_time / 60:.1f} minutes", flush=True)
    print(f"\n  Modèle sauvegardé : {save_dir}", flush=True)
    print("=" * 55, flush=True)

    return metrics


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    VIT_CROPS    = PROJECT_ROOT / "data" / "vit_crops"

    if not VIT_CROPS.exists():
        print("ERREUR: Lancez d'abord : python src/prepare_dataset.py")
        sys.exit(1)

    train_vit(
        project_root      = str(PROJECT_ROOT),
        epochs            = 10,
        batch_size        = 16,
        learning_rate     = 1e-3,
        max_train_samples = 800,
    )
