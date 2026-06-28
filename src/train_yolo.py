"""
ÉTAPE 3A : Fine-tuning de YOLOv8
==================================
YOLOv8 est un modèle de DÉTECTION d'objets.
Il détecte ET localise les objets dans l'image avec des boîtes.

Comment ça fonctionne :
- On charge YOLOv8n (nano = la version la plus légère, ~6MB)
- La couche de sortie est adaptée à nos 3 classes (Bird, Cat, Dog)
- On entraîne pendant 20 époques sur nos images
- YOLOv8 apprend à prédire : où est l'animal + quelle classe

Sorties après entraînement :
- models/yolo_finetuned/train/weights/best.pt  (meilleurs poids)
- models/yolo_finetuned/metrics_yolo.json      (métriques)
- models/yolo_finetuned/train/results.png       (courbes)
"""

import sys
import time
import json
import os
from pathlib import Path

# Ajouter le dossier parent au chemin Python pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.utils.metrics import save_metrics


def train_yolo(
    data_yaml: str,
    project_root: str,
    epochs: int = 20,
    img_size: int = 640,
    batch_size: int = 16,
) -> dict:
    """
    Fine-tune YOLOv8n sur le dataset Dogs/Cats/Birds.

    Args:
        data_yaml   : Chemin vers le fichier data.yaml
        project_root: Racine du projet
        epochs      : Nombre d'époques (tours complets du dataset)
        img_size    : Taille des images en pixels (640 = standard YOLO)
        batch_size  : Nombre d'images traitées en parallèle

    Returns:
        Dictionnaire des métriques finales
    """
    project_root = Path(project_root)
    save_dir = project_root / "models" / "yolo_finetuned"

    print("=" * 55)
    print("  FINE-TUNING YOLOV8")
    print("=" * 55)
    print(f"  Dataset : {data_yaml}")
    print(f"  Époques : {epochs}")
    print(f"  Taille image : {img_size}x{img_size}")
    print(f"  Batch size : {batch_size}")
    print()

    # ── Étape 1 : Charger le modèle pré-entraîné ──────────────────────────
    # 'yolov8n.pt' = YOLOv8 Nano, pré-entraîné sur COCO (80 classes)
    # Il sera téléchargé automatiquement la 1ère fois (~6MB)
    print("[1/3] Chargement du modèle pré-entraîné yolov8n...")
    model = YOLO("yolov8n.pt")

    # ── Étape 2 : Fine-tuning ─────────────────────────────────────────────
    # YOLOv8 adapte automatiquement la couche de sortie au nombre de classes
    print("[2/3] Début du fine-tuning...")
    start_time = time.time()

    results = model.train(
        data=data_yaml,           # Fichier de config du dataset
        epochs=epochs,            # Nombre d'époques
        imgsz=img_size,           # Taille des images
        batch=batch_size,         # Taille du batch
        project=str(save_dir),    # Dossier de sauvegarde
        name="train",             # Nom du sous-dossier
        save=True,                # Sauvegarder les checkpoints
        plots=True,               # Générer les graphiques automatiquement
        val=True,                 # Valider après chaque époque
        patience=5,               # Arrêt anticipé si pas d'amélioration après 5 époques
        device=0 if _has_gpu() else "cpu",  # GPU si disponible, sinon CPU
        verbose=True,
    )

    training_time = time.time() - start_time

    # ── Étape 3 : Évaluation finale sur le test set ───────────────────────
    print("\n[3/3] Évaluation finale sur le test set...")
    best_model_path = save_dir / "train" / "weights" / "best.pt"

    if best_model_path.exists():
        best_model = YOLO(str(best_model_path))
        test_metrics = best_model.val(data=data_yaml, split="test", verbose=False)

        precision = float(test_metrics.box.mp)   # Précision moyenne (toutes classes)
        recall    = float(test_metrics.box.mr)   # Rappel moyen
        map50     = float(test_metrics.box.map50) # mAP à IoU=0.5
        map50_95  = float(test_metrics.box.map)  # mAP à IoU=0.5:0.95

        # Calcul du F1 manuellement
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        model_size_mb = best_model_path.stat().st_size / (1024 * 1024)
    else:
        print("  ATTENTION: best.pt non trouvé, utilisation des métriques de validation")
        precision = recall = map50 = map50_95 = f1 = 0.0
        model_size_mb = 0.0

    # ── Sauvegarder les métriques ─────────────────────────────────────────
    metrics = {
        "model"                : "YOLOv8n",
        "task"                 : "détection d'objets",
        "precision"            : round(precision, 4),
        "recall"               : round(recall, 4),
        "f1"                   : round(f1, 4),
        "mAP50"                : round(map50, 4),
        "mAP50-95"             : round(map50_95, 4),
        "training_time_seconds": round(training_time, 1),
        "training_time_minutes": round(training_time / 60, 1),
        "model_size_mb"        : round(model_size_mb, 2),
        "epochs_trained"       : epochs,
        "best_model_path"      : str(best_model_path),
    }

    save_metrics(metrics, save_dir / "metrics_yolo.json")

    # ── Résumé ────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RÉSULTATS YOLOV8")
    print("=" * 55)
    print(f"  Précision   : {precision:.1%}")
    print(f"  Rappel      : {recall:.1%}")
    print(f"  F1-Score    : {f1:.1%}")
    print(f"  mAP@50      : {map50:.1%}")
    print(f"  Taille      : {model_size_mb:.1f} MB")
    print(f"  Durée entr. : {training_time / 60:.1f} minutes")
    print(f"\n  Modèle sauvegardé : {best_model_path}")
    print("=" * 55)

    return metrics


def _has_gpu() -> bool:
    """Vérifie si un GPU est disponible pour l'entraînement."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_YAML    = str(PROJECT_ROOT / "data" / "data.yaml")

    if not Path(DATA_YAML).exists():
        print("ERREUR: Le fichier data.yaml n'existe pas.")
        print("Veuillez d'abord lancer : python src/prepare_dataset.py")
        sys.exit(1)

    train_yolo(
        data_yaml    = DATA_YAML,
        project_root = str(PROJECT_ROOT),
        epochs       = 20,
        img_size     = 640,
        batch_size   = 16,
    )
