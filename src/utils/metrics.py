"""
Utilitaires pour calculer et sauvegarder les métriques de performance.

Les métriques importantes :
- Précision (Precision) : parmi toutes les détections faites, combien sont correctes ?
- Rappel (Recall)       : parmi tous les objets réels, combien ont été détectés ?
- F1-Score             : moyenne harmonique de précision et rappel (équilibre les deux)
- IoU                  : pour les masques SAM, mesure le chevauchement prédit/réel
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def compute_classification_metrics(y_true: list, y_pred: list, class_names: list) -> dict:
    """
    Calcule les métriques pour un modèle de classification (ViT).

    Args:
        y_true: Liste des vraies classes (ex: [0, 1, 2, 0, 1])
        y_pred: Liste des classes prédites
        class_names: Noms des classes (ex: ['Bird', 'Cat', 'Dog'])

    Returns:
        Dictionnaire avec precision, recall, f1, accuracy
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "num_samples": len(y_true),
        "class_names": class_names,
    }


def compute_iou_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """
    Calcule l'IoU (Intersection over Union) entre deux masques.

    IoU = aire(intersection) / aire(union)
    - IoU = 1.0 : masques identiques (parfait)
    - IoU = 0.0 : masques sans chevauchement

    Args:
        pred_mask: Masque prédit (tableau 2D booléen)
        true_mask: Masque réel (tableau 2D booléen)

    Returns:
        Score IoU entre 0 et 1
    """
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


def save_metrics(metrics: dict, save_path: str | Path) -> None:
    """
    Sauvegarde les métriques dans un fichier JSON.

    Args:
        metrics: Dictionnaire des métriques
        save_path: Chemin du fichier JSON à créer
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"  Métriques sauvegardées dans : {save_path}")


def load_metrics(model_name: str, models_dir: str | Path) -> dict | None:
    """
    Charge les métriques d'un modèle depuis son fichier JSON.

    Args:
        model_name: 'yolo', 'vit', ou 'sam'
        models_dir: Dossier racine des modèles

    Returns:
        Dictionnaire des métriques ou None si le fichier n'existe pas
    """
    models_dir = Path(models_dir)
    paths = {
        "yolo": models_dir / "yolo_finetuned" / "metrics_yolo.json",
        "vit":  models_dir / "vit_finetuned"  / "metrics_vit.json",
        "sam":  models_dir / "sam_finetuned"  / "metrics_sam.json",
    }

    path = paths.get(model_name)
    if path and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def build_comparison_table(models_dir: str | Path) -> dict:
    """
    Construit un tableau comparatif de tous les modèles disponibles.

    Returns:
        Dictionnaire avec les métriques de chaque modèle
    """
    result = {}
    for name in ["yolo", "vit", "sam"]:
        m = load_metrics(name, models_dir)
        if m:
            result[name.upper()] = m
    return result
