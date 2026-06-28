"""
ÉTAPE 1 & 2 : Préparation du dataset
=====================================
Ce script :
1. Extrait le fichier ZIP du dataset
2. Organise les données pour YOLO (format texte avec boîtes)
3. Crée des images recadrées par classe pour ViT (classification)
4. Génère le fichier data.yaml nécessaire à YOLO

Format YOLO :
  - Chaque image a un fichier texte associé
  - Chaque ligne du texte = un objet : classe cx cy largeur hauteur
  - Les coordonnées sont normalisées (entre 0 et 1)

Format ViT (classification) :
  - Un dossier par classe : Bird/, Cat/, Dog/
  - Chaque dossier contient les images de cette classe (recadrées)
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
from PIL import Image


# ---- Constantes ----
CLASS_NAMES = ["Bird", "Cat", "Dog"]


def extract_zip(zip_path: str, extract_to: str) -> Path:
    """
    Extrait le fichier ZIP dans le dossier cible.

    Le ZIP contient déjà les dossiers train/, valid/, test/
    avec des sous-dossiers images/ et labels/.
    """
    print(f"\n[1/4] Extraction du ZIP...")
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        total = len(zf.namelist())
        for i, member in enumerate(zf.namelist(), 1):
            zf.extract(member, extract_to)
            if i % 500 == 0 or i == total:
                print(f"  {i}/{total} fichiers extraits", end="\r")

    print(f"\n  Extraction terminée dans : {extract_to}")
    return extract_to


def copy_yolo_split(raw_dir: Path, dest_dir: Path, split: str) -> int:
    """
    Copie les images et labels YOLO d'un split (train/val/test)
    vers la structure finale du projet.

    Le ZIP utilise 'valid' mais notre projet utilise 'val'.
    """
    # Le ZIP utilise 'valid' pour la validation
    src_split = "valid" if split == "val" else split
    src_images = raw_dir / src_split / "images"
    src_labels = raw_dir / src_split / "labels"

    dst_images = dest_dir / split / "images"
    dst_labels = dest_dir / split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    count = 0
    if src_images.exists():
        for img_file in src_images.glob("*.jpg"):
            shutil.copy2(img_file, dst_images / img_file.name)
            # Chercher le label correspondant
            lbl_file = src_labels / img_file.with_suffix(".txt").name
            if lbl_file.exists():
                shutil.copy2(lbl_file, dst_labels / lbl_file.name)
            count += 1

    print(f"  {split}: {count} images copiées")
    return count


def create_yolo_yaml(data_dir: Path, project_root: Path) -> Path:
    """
    Crée le fichier data.yaml requis par YOLOv8 pour l'entraînement.

    Ce fichier indique à YOLO :
    - Où trouver les images (train/val/test)
    - Combien de classes il y a
    - Le nom de chaque classe
    """
    yaml_content = (
        "# Dataset pour YOLOv8 - Dogs, Cats & Birds\n"
        f"path: {data_dir.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "nc: 3\n"
        "names: ['Bird', 'Cat', 'Dog']\n"
    )
    yaml_path = data_dir / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"  data.yaml créé : {yaml_path}")
    return yaml_path


def crop_objects_for_vit(data_dir: Path, vit_dir: Path, split: str, min_size: int = 32) -> int:
    """
    Recadre les objets à partir des boîtes YOLO et les organise par classe.
    Ces images recadrées seront utilisées pour entraîner ViT.

    Exemple de transformation :
    - Image originale : photo de jardin avec un oiseau et un chat
    - Résultat : 2 images recadrées sauvegardées dans Bird/ et Cat/

    Args:
        data_dir: Dossier contenant les images et labels YOLO
        vit_dir: Dossier de destination pour ViT (classifié par dossier)
        split: 'train', 'val', ou 'test'
        min_size: Taille minimale en pixels pour garder un crop (évite les petits objets)
    """
    images_dir = data_dir / split / "images"
    labels_dir = data_dir / split / "labels"

    if not images_dir.exists():
        return 0

    count = 0
    skipped = 0

    for img_path in images_dir.glob("*.jpg"):
        lbl_path = labels_dir / img_path.with_suffix(".txt").name
        if not lbl_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            W, H = img.size

            # Lire chaque ligne du fichier label
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                # Format YOLO : classe cx cy w h (normalisé)
                class_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                # Convertir en coordonnées absolues en pixels
                x1 = int((cx - bw / 2) * W)
                y1 = int((cy - bh / 2) * H)
                x2 = int((cx + bw / 2) * W)
                y2 = int((cy + bh / 2) * H)

                # Clamp les valeurs pour rester dans l'image
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                crop_w, crop_h = x2 - x1, y2 - y1
                if crop_w < min_size or crop_h < min_size:
                    skipped += 1
                    continue

                # Recadrer l'objet
                crop = img.crop((x1, y1, x2, y2))

                # Sauvegarder dans le bon dossier de classe
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
                save_dir = vit_dir / split / class_name
                save_dir.mkdir(parents=True, exist_ok=True)

                # Nom unique pour éviter les conflits
                save_name = f"{img_path.stem}_obj{count}.jpg"
                crop.save(save_dir / save_name, quality=90)
                count += 1

        except Exception as e:
            print(f"  Erreur sur {img_path.name}: {e}")

    print(f"  ViT {split}: {count} crops créés ({skipped} trop petits, ignorés)")
    return count


def prepare_all(zip_path: str, project_root: str) -> None:
    """
    Fonction principale : prépare tout le dataset.
    À appeler une seule fois avant de lancer les entraînements.
    """
    project_root = Path(project_root)
    raw_dir = project_root / "data" / "raw"
    data_dir = project_root / "data"
    vit_dir = project_root / "data" / "vit_crops"

    print("=" * 55)
    print("  PRÉPARATION DU DATASET - Dogs, Cats & Birds")
    print("=" * 55)

    # 1. Extraire le ZIP
    extract_zip(zip_path, raw_dir)

    # 2. Copier les données YOLO
    print("\n[2/4] Organisation des données YOLO...")
    for split in ["train", "val", "test"]:
        copy_yolo_split(raw_dir, data_dir, split)

    # 3. Créer le fichier YAML pour YOLO
    print("\n[3/4] Création de data.yaml...")
    create_yolo_yaml(data_dir, project_root)

    # 4. Créer les crops pour ViT
    print("\n[4/4] Création des images recadrées pour ViT...")
    for split in ["train", "val", "test"]:
        crop_objects_for_vit(data_dir, vit_dir, split)

    print("\n" + "=" * 55)
    print("  Dataset prêt ! Structure créée :")
    print(f"  data/train/images  -> images YOLO train")
    print(f"  data/val/images    -> images YOLO validation")
    print(f"  data/test/images   -> images YOLO test")
    print(f"  data/vit_crops/    -> images par classe pour ViT")
    print(f"  data/data.yaml     -> config YOLO")
    print("=" * 55)
    print("\nProchaine étape : lancer les scripts d'entraînement !")
    print("  python src/train_yolo.py")
    print("  python src/train_vit.py")
    print("  python src/train_sam.py")


if __name__ == "__main__":
    # Chemin du fichier ZIP (modifiez si nécessaire)
    ZIP_PATH = r"C:\Users\aitla\Downloads\Dogs- Cats and Birds.v1i.yolov8.zip"
    PROJECT_ROOT = Path(__file__).parent.parent

    prepare_all(ZIP_PATH, str(PROJECT_ROOT))
