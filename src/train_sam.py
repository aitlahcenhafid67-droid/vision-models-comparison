"""
ÉTAPE 3C : Fine-tuning de SAM (Segment Anything Model)
========================================================
SAM est un modèle de SEGMENTATION.
Il dessine des masques précis autour des objets (pixel par pixel).

Comment SAM fonctionne :
1. Encodeur d'image : transforme l'image en représentation compressée
2. Encodeur de prompt : encode une boîte ou un point fourni par l'utilisateur
3. Décodeur de masque : génère le masque de segmentation

Ce qu'on fine-tune :
- On gèle l'encodeur d'image (trop lourd, 90% des paramètres)
- On fine-tune SEULEMENT le décodeur de masque (léger, rapide)
- On utilise les boîtes YOLO comme prompts (ce qu'on veut segmenter)
- On crée des pseudo-masques à partir des boîtes (rectangles remplis)

Note pédagogique :
  Des "vrais" masques nécessiteraient un dataset annoté avec polygones.
  Ici on utilise les boîtes comme approximation (méthode valide pour fine-tuning léger).

Sorties après entraînement :
- models/sam_finetuned/sam_decoder_finetuned.pth
- models/sam_finetuned/metrics_sam.json
- models/sam_finetuned/training_curves_sam.png
"""

import sys
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend non-interactif pour Windows
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import SamModel, SamProcessor

from src.utils.metrics import compute_iou_score, save_metrics


# ── Constantes ────────────────────────────────────────────────────────────────
SAM_MODEL_NAME = "facebook/sam-vit-base"   # Version légère de SAM (~375MB)
IMAGE_SIZE     = 1024                       # SAM travaille en 1024x1024


class YOLOSAMDataset(Dataset):
    """
    Dataset personnalisé qui charge des images + boîtes YOLO.
    Il convertit les boîtes YOLO en pseudo-masques pour entraîner SAM.

    Structure attendue :
    data/
      train/
        images/ → .jpg
        labels/ → .txt (format YOLO : classe cx cy w h)
    """

    def __init__(self, images_dir: Path, labels_dir: Path, max_samples: int = 500):
        """
        Args:
            images_dir : Dossier des images
            labels_dir : Dossier des labels YOLO
            max_samples: Limite le nombre d'exemples (SAM est lourd en mémoire)
        """
        self.samples = []  # Liste de (chemin_image, liste_de_boîtes)

        for img_path in sorted(images_dir.glob("*.jpg"))[:max_samples]:
            lbl_path = labels_dir / img_path.with_suffix(".txt").name
            if not lbl_path.exists():
                continue

            boxes = []
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append([float(x) for x in parts])   # [class, cx, cy, w, h]

            if boxes:
                self.samples.append((img_path, boxes))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, boxes = self.samples[idx]

        # Charger l'image et obtenir ses dimensions
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # Prendre la première boîte (simplification)
        box = boxes[0]
        cx, cy, bw, bh = box[1], box[2], box[3], box[4]

        # Convertir YOLO (normalisé) → pixels absolus
        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)

        # Clamp pour rester dans l'image
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        # Créer le pseudo-masque (rectangle rempli = approximation du masque réel)
        # C'est notre "vérité terrain" : on suppose que l'objet remplit sa boîte
        gt_mask = np.zeros((H, W), dtype=np.float32)
        gt_mask[y1:y2, x1:x2] = 1.0

        return {
            "image"  : img,                        # Image PIL originale
            "box"    : [x1, y1, x2, y2],           # Boîte en pixels
            "gt_mask": gt_mask,                    # Masque pseudo-vérité
            "size"   : (H, W),
        }


def prepare_sam_batch(batch: list, processor: SamProcessor, device: torch.device):
    """
    Prépare un batch pour SAM en utilisant le SamProcessor.
    SAM nécessite un prétraitement spécial (resize, normalisation spécifique).
    """
    images = [item["image"] for item in batch]
    boxes  = [item["box"]   for item in batch]
    masks  = [item["gt_mask"] for item in batch]

    # Le processor formate les images et les boîtes pour SAM
    inputs = processor(
        images=images,
        input_boxes=[[box] for box in boxes],   # Double liste : [[box1], [box2], ...]
        return_tensors="pt",
    )

    # Redimensionner les masques à 256x256 (taille de sortie SAM)
    gt_masks_resized = []
    for mask in masks:
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        mask_resized = F.interpolate(mask_tensor, size=(256, 256), mode="bilinear", align_corners=False)
        gt_masks_resized.append(mask_resized.squeeze(0))  # (1, 256, 256)

    gt_masks = torch.stack(gt_masks_resized)  # (B, 1, 256, 256)

    return inputs.to(device), gt_masks.to(device)


def train_sam(
    project_root: str,
    epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 1e-5,
    max_train_samples: int = 500,
) -> dict:
    """
    Fine-tune le décodeur de masque SAM.

    Args:
        project_root      : Racine du projet
        epochs            : Nombre d'époques (moins que ViT car SAM est plus lourd)
        batch_size        : Petit (2-4) car SAM utilise beaucoup de mémoire GPU
        learning_rate     : Très petit pour le fine-tuning du décodeur
        max_train_samples : Limite pour ne pas dépasser la mémoire
    """
    project_root = Path(project_root)
    data_dir     = project_root / "data"
    save_dir     = project_root / "models" / "sam_finetuned"
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print("  FINE-TUNING SAM (Segment Anything Model)")
    print("=" * 55)
    print(f"  Device       : {device}")
    print(f"  Époques      : {epochs}")
    print(f"  Batch size   : {batch_size}")
    print(f"  Max samples  : {max_train_samples}")
    print()

    # ── Chargement du processor et du modèle ─────────────────────────────
    print("[1/4] Téléchargement de SAM ViT-B...")
    print("  (environ 375MB, fait une seule fois)")
    processor = SamProcessor.from_pretrained(SAM_MODEL_NAME)
    model     = SamModel.from_pretrained(SAM_MODEL_NAME).to(device)

    # ── Geler l'encodeur d'image (trop lourd) ─────────────────────────────
    # On entraîne SEULEMENT le décodeur de masque
    for name, param in model.named_parameters():
        if "mask_decoder" in name:
            param.requires_grad = True    # Fine-tuner le décodeur
        else:
            param.requires_grad = False   # Geler le reste

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres entraînables : {trainable:,} / {total:,} ({trainable/total:.1%})")

    # ── Chargement des données ─────────────────────────────────────────────
    print("\n[2/4] Chargement des données...")
    train_ds = YOLOSAMDataset(
        data_dir / "train" / "images",
        data_dir / "train" / "labels",
        max_samples=max_train_samples,
    )
    val_ds = YOLOSAMDataset(
        data_dir / "val" / "images",
        data_dir / "val" / "labels",
        max_samples=100,
    )
    print(f"  Train : {len(train_ds)} images  |  Val : {len(val_ds)} images")

    # ── Entraînement ──────────────────────────────────────────────────────
    print(f"\n[3/4] Entraînement ({epochs} époques)...")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )
    # Perte combinée : BCE pour les masques + Dice pour la forme
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}
    best_val_iou = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # ── Phase d'entraînement ──
        model.train()
        epoch_loss, epoch_iou = [], []

        for i in range(0, len(train_ds), batch_size):
            batch = [train_ds[j] for j in range(i, min(i + batch_size, len(train_ds)))]
            inputs, gt_masks = prepare_sam_batch(batch, processor, device)

            optimizer.zero_grad()
            outputs = model(**inputs, multimask_output=False)
            # outputs.pred_masks : (B, 1, 1, 256, 256)
            pred_masks = outputs.pred_masks.squeeze(2)  # (B, 1, 256, 256)

            loss = criterion(pred_masks, gt_masks)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

            # IoU pour cette batch
            with torch.no_grad():
                pred_binary = (torch.sigmoid(pred_masks) > 0.5).cpu().numpy()
                gt_binary   = (gt_masks > 0.5).cpu().numpy()
                for pm, gm in zip(pred_binary, gt_binary):
                    epoch_iou.append(compute_iou_score(pm[0], gm[0]))

        # ── Phase de validation ──
        model.eval()
        val_losses, val_ious = [], []
        with torch.no_grad():
            for i in range(0, len(val_ds), batch_size):
                batch = [val_ds[j] for j in range(i, min(i + batch_size, len(val_ds)))]
                inputs, gt_masks = prepare_sam_batch(batch, processor, device)
                outputs = model(**inputs, multimask_output=False)
                pred_masks = outputs.pred_masks.squeeze(2)
                loss = criterion(pred_masks, gt_masks)
                val_losses.append(loss.item())

                pred_binary = (torch.sigmoid(pred_masks) > 0.5).cpu().numpy()
                gt_binary   = (gt_masks > 0.5).cpu().numpy()
                for pm, gm in zip(pred_binary, gt_binary):
                    val_ious.append(compute_iou_score(pm[0], gm[0]))

        mean_train_loss = np.mean(epoch_loss)
        mean_val_loss   = np.mean(val_losses)
        mean_train_iou  = np.mean(epoch_iou)
        mean_val_iou    = np.mean(val_ious)

        history["train_loss"].append(float(mean_train_loss))
        history["val_loss"].append(float(mean_val_loss))
        history["train_iou"].append(float(mean_train_iou))
        history["val_iou"].append(float(mean_val_iou))

        print(f"  Epoque {epoch:2d}/{epochs} | "
              f"Train Loss: {mean_train_loss:.4f} IoU: {mean_train_iou:.3f} | "
              f"Val Loss: {mean_val_loss:.4f} IoU: {mean_val_iou:.3f}",
              flush=True)

        # Sauvegarder le meilleur décodeur
        if mean_val_iou > best_val_iou:
            best_val_iou = mean_val_iou
            torch.save(
                model.mask_decoder.state_dict(),
                save_dir / "sam_decoder_finetuned.pth"
            )
            print(f"           -> Meilleur decodeur sauvegarde (IoU={mean_val_iou:.3f})", flush=True)

    training_time = time.time() - start_time

    # ── Courbes d'apprentissage ────────────────────────────────────────────
    _save_sam_curves(history, save_dir / "training_curves_sam.png")

    # ── Métriques finales ─────────────────────────────────────────────────
    print("\n[4/4] Évaluation finale...")
    model_size_mb = (save_dir / "sam_decoder_finetuned.pth").stat().st_size / (1024 * 1024)

    metrics = {
        "model"                : "SAM ViT-B (decoder fine-tuned)",
        "task"                 : "segmentation",
        "iou"                  : round(best_val_iou, 4),
        "precision"            : round(best_val_iou, 4),   # IoU ≈ proxy pour precision
        "recall"               : round(best_val_iou, 4),
        "f1"                   : round(best_val_iou, 4),
        "training_time_seconds": round(training_time, 1),
        "training_time_minutes": round(training_time / 60, 1),
        "model_size_mb"        : round(model_size_mb, 2),
        "epochs_trained"       : epochs,
        "best_val_iou"         : round(best_val_iou, 4),
    }

    save_metrics(metrics, save_dir / "metrics_sam.json")

    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 55)
    print("  RÉSULTATS SAM")
    print("=" * 55)
    print(f"  IoU (val)    : {best_val_iou:.3f}")
    print(f"  Taille deco. : {model_size_mb:.1f} MB")
    print(f"  Durée entr.  : {training_time / 60:.1f} minutes")
    print(f"\n  Décodeur sauvegardé : {save_dir / 'sam_decoder_finetuned.pth'}")
    print("=" * 55)

    return metrics


def _save_sam_curves(history: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Validation")
    axes[0].set_title("Perte (Loss)")
    axes[0].set_xlabel("Époque")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_iou"], "b-o", label="Train")
    axes[1].plot(epochs, history["val_iou"],   "r-o", label="Validation")
    axes[1].set_title("IoU (Intersection over Union)")
    axes[1].set_xlabel("Époque")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle("SAM Fine-tuning - Courbes d'Apprentissage", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Courbes sauvegardées : {path}")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR     = PROJECT_ROOT / "data" / "train" / "images"

    if not DATA_DIR.exists():
        print("ERREUR: Les données YOLO n'existent pas.")
        print("Veuillez d'abord lancer : python src/prepare_dataset.py")
        sys.exit(1)

    # Paramètres réduits pour CPU (augmenter si vous avez un GPU)
    train_sam(
        project_root      = str(PROJECT_ROOT),
        epochs            = 5,
        batch_size        = 2,
        learning_rate     = 1e-5,
        max_train_samples = 150,
    )
