"""
Utilitaires de visualisation : dessiner les résultats sur les images.

Chaque modèle produit un type de sortie différent :
- YOLO    → boîtes englobantes (rectangles) avec étiquettes
- ViT     → étiquette de classe + barre de confiance
- SAM     → masques colorés superposés à l'image
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# Couleurs par classe : Bird=vert, Cat=bleu, Dog=rouge
CLASS_COLORS = {
    0: (0, 200, 0),    # Bird - vert
    1: (0, 100, 255),  # Cat  - bleu
    2: (220, 50, 50),  # Dog  - rouge
}
CLASS_NAMES = {0: "Bird", 1: "Cat", 2: "Dog"}


def draw_yolo_boxes(image: Image.Image, boxes: list, scores: list, class_ids: list, class_names: dict | None = None) -> Image.Image:
    """
    Dessine les boîtes YOLO sur l'image.

    Args:
        image: Image PIL originale
        boxes: Liste de [x1, y1, x2, y2] (coordonnées absolues en pixels)
        scores: Liste des scores de confiance [0.0 - 1.0]
        class_ids: Liste des IDs de classe (0=Bird, 1=Cat, 2=Dog)

    Returns:
        Image avec les boîtes dessinées
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = [int(c) for c in box]
        names = class_names if class_names is not None else CLASS_NAMES
        color = CLASS_COLORS.get(cid % 3, (255, 255, 0))
        name = names.get(cid, f"Class {cid}")

        # Rectangle de la boîte (épaisseur 3)
        for thickness in range(3):
            draw.rectangle(
                [x1 - thickness, y1 - thickness, x2 + thickness, y2 + thickness],
                outline=color
            )

        # Étiquette avec le nom et le score
        label = f"{name}: {score:.1%}"
        label_y = max(y1 - 22, 0)
        draw.rectangle([x1, label_y, x1 + len(label) * 7 + 4, label_y + 18], fill=color)
        draw.text((x1 + 2, label_y + 2), label, fill=(255, 255, 255))

    return img


def draw_vit_result(image: Image.Image, class_name: str, confidence: float) -> Image.Image:
    """
    Affiche le résultat de classification ViT sur l'image.

    Args:
        image: Image PIL originale
        class_name: Nom de la classe prédite (ex: "Dog")
        confidence: Score de confiance entre 0 et 1

    Returns:
        Image avec l'étiquette superposée
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Fond semi-transparent en bas de l'image
    overlay_h = 50
    draw.rectangle([0, h - overlay_h, w, h], fill=(0, 0, 0, 180))

    # Texte de la prédiction
    text = f"{class_name}  {confidence:.1%}"
    draw.text((10, h - overlay_h + 10), text, fill=(255, 255, 255))

    # Barre de confiance
    bar_w = int((w - 20) * confidence)
    draw.rectangle([10, h - 15, 10 + bar_w, h - 5], fill=(0, 200, 100))

    return img


def draw_sam_masks(image: Image.Image, masks: list, class_ids: list | None = None) -> Image.Image:
    """
    Superpose les masques SAM sur l'image avec des couleurs semi-transparentes.

    Args:
        image: Image PIL originale
        masks: Liste de tableaux numpy 2D (booléens)
        class_ids: IDs de classe pour choisir la couleur (optionnel)

    Returns:
        Image avec les masques colorés
    """
    img_array = np.array(image.convert("RGBA"))

    colors = [
        (0, 200, 0, 120),    # vert
        (0, 100, 255, 120),  # bleu
        (220, 50, 50, 120),  # rouge
        (255, 200, 0, 120),  # jaune
        (200, 0, 255, 120),  # violet
    ]

    for i, mask in enumerate(masks):
        if mask is None or mask.sum() == 0:
            continue

        # Choisir la couleur selon la classe ou l'index
        if class_ids and i < len(class_ids):
            color = CLASS_COLORS.get(class_ids[i], (255, 165, 0))
            rgba_color = color + (120,)
        else:
            rgba_color = colors[i % len(colors)]

        # Appliquer la couleur sur les pixels du masque
        overlay = np.zeros_like(img_array)
        overlay[mask > 0] = rgba_color

        # Mélanger l'overlay avec l'image originale
        alpha = rgba_color[3] / 255.0
        img_array = np.where(
            (overlay[..., 3:4] > 0),
            (img_array * (1 - alpha) + overlay * alpha).astype(np.uint8),
            img_array
        )

        # Contour du masque avec PIL (sans cv2)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        edges = np.array(mask_pil.filter(ImageFilter.FIND_EDGES)) > 0
        img_array[edges] = list(rgba_color[:3]) + [255]

    return Image.fromarray(img_array, "RGBA").convert("RGB")
