"""
APPLICATION STREAMLIT - Comparaison de Modèles Vision
=======================================================
Cette application permet de :
1. Charger une image et tester les 3 modèles fine-tunés
2. Visualiser les résultats (boîtes YOLO, classe ViT, masques SAM)
3. Comparer les métriques et les courbes d'apprentissage

Lancement :
    streamlit run app.py
"""

import os
import sys
import time
import json
from pathlib import Path

# Ajouter le dossier du projet au chemin Python
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Imports conditionnels (évite les crashs si une lib n'est pas installée)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.metrics import load_metrics, build_comparison_table
from src.utils.visualize import draw_yolo_boxes, draw_vit_result, draw_sam_masks, CLASS_NAMES

# ── Configuration de la page ────────────────────────────────────────────────
st.set_page_config(
    page_title="Vision Model Comparison",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Chemins importants
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR   = PROJECT_ROOT / "models"


# ════════════════════════════════════════════════════════════════════════════
# FONCTIONS DE CHARGEMENT DES MODÈLES
# (utilise st.session_state pour ne charger qu'une fois)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Chargement de YOLOv8...")
def load_yolo_model():
    """
    Charge le modèle YOLOv8 fine-tuné.
    @st.cache_resource garde le modèle en mémoire entre les requêtes.
    """
    from ultralytics import YOLO

    best_pt = MODELS_DIR / "yolo_finetuned" / "train" / "weights" / "best.pt"

    if best_pt.exists():
        model = YOLO(str(best_pt))
        return model, "fine-tuné", str(best_pt)
    else:
        # Fallback : modèle de base (pas fine-tuné)
        model = YOLO("yolov8n.pt")
        return model, "base (non fine-tuné)", "yolov8n.pt"


@st.cache_resource(show_spinner="Chargement de ViT...")
def load_vit_model():
    """Charge le modèle ViT fine-tuné."""
    from transformers import ViTForImageClassification, ViTImageProcessor

    save_dir = MODELS_DIR / "vit_finetuned"
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if save_dir.exists() and (save_dir / "config.json").exists():
        model     = ViTForImageClassification.from_pretrained(str(save_dir)).to(device)
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        return model, processor, device, "fine-tuné"
    else:
        # Fallback : modèle de base
        model     = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", num_labels=3, ignore_mismatched_sizes=True
        ).to(device)
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        return model, processor, device, "base (non fine-tuné)"


@st.cache_resource(show_spinner="Chargement de SAM...")
def load_sam_model():
    """Charge SAM avec le décodeur fine-tuné si disponible."""
    from transformers import SamModel, SamProcessor

    save_dir     = MODELS_DIR / "sam_finetuned"
    decoder_path = save_dir / "sam_decoder_finetuned.pth"
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model     = SamModel.from_pretrained("facebook/sam-vit-base").to(device)

    if decoder_path.exists():
        state_dict = torch.load(str(decoder_path), map_location=device)
        model.mask_decoder.load_state_dict(state_dict)
        status = "fine-tuné"
    else:
        status = "base (non fine-tuné)"

    model.eval()
    return model, processor, device, status


# ════════════════════════════════════════════════════════════════════════════
# FONCTIONS D'INFÉRENCE
# ════════════════════════════════════════════════════════════════════════════

def run_yolo_inference(image: Image.Image) -> dict:
    """
    Lance la détection YOLOv8 sur une image.

    Returns:
        dict avec boxes, scores, class_ids, inference_time, result_image
    """
    model, status, _ = load_yolo_model()

    start = time.time()
    results = model(image, verbose=False)
    elapsed = time.time() - start

    # Extraire les résultats
    boxes, scores, class_ids = [], [], []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            boxes.append(box.xyxy[0].cpu().tolist())   # [x1, y1, x2, y2]
            scores.append(float(box.conf[0]))
            class_ids.append(int(box.cls[0]))

    # Dessiner les boîtes sur l'image
    result_image = draw_yolo_boxes(image, boxes, scores, class_ids)

    return {
        "boxes"         : boxes,
        "scores"        : scores,
        "class_ids"     : class_ids,
        "inference_time": elapsed,
        "result_image"  : result_image,
        "status"        : status,
        "num_detections": len(boxes),
    }


def run_vit_inference(image: Image.Image) -> dict:
    """
    Lance la classification ViT sur une image.

    Returns:
        dict avec class_name, confidence, all_probs, inference_time
    """
    model, processor, device, status = load_vit_model()

    # Prétraiter l'image pour ViT
    inputs = processor(images=image, return_tensors="pt").to(device)

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    elapsed = time.time() - start

    # Convertir les logits en probabilités
    probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().tolist()

    # Classe prédite = celle avec la probabilité la plus haute
    top_idx    = int(np.argmax(probs))
    top_class  = CLASS_NAMES.get(top_idx, f"Class {top_idx}")
    confidence = probs[top_idx]

    # Dessiner le résultat sur l'image
    result_image = draw_vit_result(image, top_class, confidence)

    return {
        "class_name"    : top_class,
        "class_id"      : top_idx,
        "confidence"    : confidence,
        "all_probs"     : {CLASS_NAMES[i]: p for i, p in enumerate(probs) if i in CLASS_NAMES},
        "inference_time": elapsed,
        "result_image"  : result_image,
        "status"        : status,
    }


def run_sam_inference(image: Image.Image, input_box: list | None = None) -> dict:
    """
    Lance la segmentation SAM sur une image.

    Args:
        image    : Image PIL
        input_box: [x1, y1, x2, y2] boîte de prompt (optionnel)
                   Si None, utilise le centre de l'image

    Returns:
        dict avec masks, iou_scores, inference_time, result_image
    """
    model, processor, device, status = load_sam_model()

    W, H = image.size

    # Si aucune boîte fournie, utiliser un cadre centré (80% de l'image)
    if input_box is None:
        margin_x, margin_y = int(W * 0.1), int(H * 0.1)
        input_box = [margin_x, margin_y, W - margin_x, H - margin_y]

    # Préparer les inputs pour SAM
    inputs = processor(
        images=image,
        input_boxes=[[input_box]],
        return_tensors="pt",
    ).to(device)

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    elapsed = time.time() - start

    # Extraire les masques (résolution originale)
    masks_tensor = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]  # (1, 1, H, W)

    masks      = [(masks_tensor[0, 0].numpy() > 0.5)]
    iou_scores = outputs.iou_scores[0].cpu().tolist() if hasattr(outputs, "iou_scores") else [0.0]

    # Dessiner les masques
    result_image = draw_sam_masks(image, masks)

    return {
        "masks"         : masks,
        "iou_scores"    : iou_scores,
        "inference_time": elapsed,
        "result_image"  : result_image,
        "input_box"     : input_box,
        "status"        : status,
    }


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

def show_sidebar():
    """Affiche la barre latérale avec les informations du projet."""
    with st.sidebar:
        st.title("Vision Models")
        st.markdown("**Comparaison de modèles de vision**")
        st.divider()

        st.subheader("Dataset")
        st.markdown("""
        - **Bird** (oiseau)
        - **Cat** (chat)
        - **Dog** (chien)
        - 4 784 images (train+val+test)
        """)

        st.divider()
        st.subheader("Modèles")

        # Afficher l'état de chaque modèle
        for name, label in [("yolo", "YOLOv8 (détection)"),
                             ("vit",  "ViT (classification)"),
                             ("sam",  "SAM (segmentation)")]:
            metrics_file = {
                "yolo": MODELS_DIR / "yolo_finetuned" / "metrics_yolo.json",
                "vit":  MODELS_DIR / "vit_finetuned"  / "metrics_vit.json",
                "sam":  MODELS_DIR / "sam_finetuned"  / "metrics_sam.json",
            }[name]

            if metrics_file.exists():
                st.success(f"{label}")
            else:
                st.warning(f"{label} (non entraîné)")

        st.divider()
        st.caption("OFPPT - TS DIA-IA-102\nFormateur : Mr. SABER")


# ════════════════════════════════════════════════════════════════════════════
# ONGLET INFÉRENCE
# ════════════════════════════════════════════════════════════════════════════

def tab_inference():
    """Onglet principal pour tester les modèles sur une image."""
    st.header("Inférence — Tester un modèle")

    # ── Upload d'image ──────────────────────────────────────────────────────
    col_upload, col_options = st.columns([1, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Charger une image (JPEG ou PNG)",
            type=["jpg", "jpeg", "png"],
            help="Choisissez une image contenant un oiseau, un chat ou un chien"
        )

    with col_options:
        if uploaded:
            model_choice = st.selectbox(
                "Choisir le modèle",
                ["YOLOv8 (Détection)", "ViT (Classification)", "SAM (Segmentation)", "Tous les modèles"],
            )
            run_btn = st.button("Lancer l'inférence", type="primary", use_container_width=True)

    if not uploaded:
        st.info("Veuillez charger une image pour commencer.")
        # Montrer des exemples depuis le dataset
        _show_dataset_samples()
        return

    image = Image.open(uploaded).convert("RGB")

    st.divider()

    if not run_btn:
        st.image(image, caption="Image chargée", use_container_width=True)
        return

    # ── Exécuter l'inférence ────────────────────────────────────────────────
    if model_choice == "Tous les modèles":
        _run_all_models(image)
    elif "YOLOv8" in model_choice:
        _show_yolo_result(image)
    elif "ViT" in model_choice:
        _show_vit_result(image)
    elif "SAM" in model_choice:
        _show_sam_result(image)


def _show_yolo_result(image: Image.Image):
    with st.spinner("Détection YOLO en cours..."):
        result = run_yolo_inference(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image originale")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader(f"Résultat YOLO ({result['status']})")
        st.image(result["result_image"], use_container_width=True)

    # Métriques
    c1, c2, c3 = st.columns(3)
    c1.metric("Objets détectés", result["num_detections"])
    c2.metric("Temps d'inférence", f"{result['inference_time']*1000:.1f} ms")
    if result["scores"]:
        c3.metric("Confiance max.", f"{max(result['scores']):.1%}")

    # Tableau des détections
    if result["boxes"]:
        st.subheader("Détails des détections")
        rows = []
        for box, score, cid in zip(result["boxes"], result["scores"], result["class_ids"]):
            rows.append({
                "Classe": CLASS_NAMES.get(cid, f"Class {cid}"),
                "Confiance": f"{score:.1%}",
                "x1": int(box[0]), "y1": int(box[1]),
                "x2": int(box[2]), "y2": int(box[3]),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _show_vit_result(image: Image.Image):
    with st.spinner("Classification ViT en cours..."):
        result = run_vit_inference(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image originale")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader(f"Résultat ViT ({result['status']})")
        st.image(result["result_image"], use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Classe prédite", result["class_name"])
    c2.metric("Confiance", f"{result['confidence']:.1%}")
    st.metric("Temps d'inférence", f"{result['inference_time']*1000:.1f} ms")

    # Graphique des probabilités
    if result["all_probs"]:
        st.subheader("Probabilités par classe")
        probs_df = pd.DataFrame(list(result["all_probs"].items()), columns=["Classe", "Probabilité"])
        fig = px.bar(probs_df, x="Classe", y="Probabilité",
                     color="Probabilité", color_continuous_scale="Blues",
                     range_y=[0, 1])
        st.plotly_chart(fig, use_container_width=True)


def _show_sam_result(image: Image.Image, input_box=None):
    with st.spinner("Segmentation SAM en cours..."):
        result = run_sam_inference(image, input_box)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image originale")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader(f"Résultat SAM ({result['status']})")
        st.image(result["result_image"], use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Masques générés", len(result["masks"]))
    c2.metric("Temps d'inférence", f"{result['inference_time']*1000:.1f} ms")


def _run_all_models(image: Image.Image):
    """Exécute les 3 modèles en parallèle et affiche les résultats côte à côte."""
    st.subheader("Comparaison des 3 modèles")

    # Exécuter les 3 modèles
    with st.spinner("Exécution des 3 modèles..."):
        try:
            yolo_r = run_yolo_inference(image)
        except Exception as e:
            yolo_r = {"error": str(e)}

        try:
            vit_r = run_vit_inference(image)
        except Exception as e:
            vit_r = {"error": str(e)}

        try:
            # Utiliser les boîtes YOLO comme prompt SAM si disponibles
            sam_box = None
            if "boxes" in yolo_r and yolo_r["boxes"]:
                sam_box = [int(x) for x in yolo_r["boxes"][0]]
            sam_r = run_sam_inference(image, sam_box)
        except Exception as e:
            sam_r = {"error": str(e)}

    # Afficher les résultats en 3 colonnes
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### YOLO — Détection")
        if "error" in yolo_r:
            st.error(f"Erreur: {yolo_r['error']}")
        else:
            st.image(yolo_r["result_image"], use_container_width=True)
            st.metric("Objets détectés", yolo_r["num_detections"])
            st.metric("Temps", f"{yolo_r['inference_time']*1000:.1f} ms")

    with col2:
        st.markdown("### ViT — Classification")
        if "error" in vit_r:
            st.error(f"Erreur: {vit_r['error']}")
        else:
            st.image(vit_r["result_image"], use_container_width=True)
            st.metric("Classe", vit_r["class_name"])
            st.metric("Confiance", f"{vit_r['confidence']:.1%}")
            st.metric("Temps", f"{vit_r['inference_time']*1000:.1f} ms")

    with col3:
        st.markdown("### SAM — Segmentation")
        if "error" in sam_r:
            st.error(f"Erreur: {sam_r['error']}")
        else:
            st.image(sam_r["result_image"], use_container_width=True)
            st.metric("Masques", len(sam_r["masks"]))
            st.metric("Temps", f"{sam_r['inference_time']*1000:.1f} ms")


def _show_dataset_samples():
    """Affiche quelques exemples du dataset si les images sont disponibles."""
    test_dir = PROJECT_ROOT / "data" / "test" / "images"
    if not test_dir.exists():
        return

    images = list(test_dir.glob("*.jpg"))[:6]
    if not images:
        return

    st.subheader("Exemples du dataset test")
    cols = st.columns(3)
    for i, img_path in enumerate(images[:6]):
        with cols[i % 3]:
            st.image(str(img_path), use_container_width=True, caption=img_path.name[:20])


# ════════════════════════════════════════════════════════════════════════════
# ONGLET COMPARAISON
# ════════════════════════════════════════════════════════════════════════════

def tab_comparison():
    """Onglet de comparaison des métriques entre les modèles."""
    st.header("Comparaison des Modèles")

    comparison = build_comparison_table(MODELS_DIR)

    if not comparison:
        st.warning(
            "Aucun modèle n'a encore été entraîné.\n\n"
            "Veuillez lancer les scripts d'entraînement :\n"
            "```bash\n"
            "python src/prepare_dataset.py\n"
            "python src/train_yolo.py\n"
            "python src/train_vit.py\n"
            "python src/train_sam.py\n"
            "```"
        )
        return

    # ── Tableau comparatif des métriques ───────────────────────────────────
    st.subheader("Tableau Comparatif des Métriques")

    rows = []
    for model_name, m in comparison.items():
        rows.append({
            "Modèle"       : model_name,
            "Tâche"        : m.get("task", "-"),
            "Précision"    : f"{m.get('precision', 0):.1%}",
            "Rappel"       : f"{m.get('recall', 0):.1%}",
            "F1-Score"     : f"{m.get('f1', 0):.1%}",
            "Métrique clé" : _get_key_metric(model_name, m),
            "Taille (MB)"  : f"{m.get('model_size_mb', 0):.1f}",
            "Entr. (min)"  : f"{m.get('training_time_minutes', 0):.1f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Graphiques comparatifs ─────────────────────────────────────────────
    st.subheader("Graphiques Comparatifs")
    col1, col2 = st.columns(2)

    with col1:
        _plot_metrics_bar(comparison)

    with col2:
        _plot_model_size_latency(comparison)

    # ── Courbes d'apprentissage ────────────────────────────────────────────
    st.subheader("Courbes d'Apprentissage")
    _show_training_curves()

    # ── Analyse ────────────────────────────────────────────────────────────
    st.subheader("Analyse des Forces et Faiblesses")
    _show_model_analysis()


def _get_key_metric(model_name: str, metrics: dict) -> str:
    """Retourne la métrique principale selon le modèle."""
    if model_name == "YOLO":
        return f"mAP50: {metrics.get('mAP50', 0):.1%}"
    elif model_name == "VIT":
        return f"Accuracy: {metrics.get('accuracy', 0):.1%}"
    elif model_name == "SAM":
        return f"IoU: {metrics.get('iou', 0):.3f}"
    return "-"


def _plot_metrics_bar(comparison: dict):
    """Graphique en barres des métriques F1."""
    fig = go.Figure()

    models  = list(comparison.keys())
    metrics = ["precision", "recall", "f1"]
    colors  = ["#2196F3", "#4CAF50", "#FF5722"]

    for metric, color in zip(metrics, colors):
        values = [comparison[m].get(metric, 0) for m in models]
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=models,
            y=values,
            marker_color=color,
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
        ))

    fig.update_layout(
        title="Précision / Rappel / F1 par modèle",
        barmode="group",
        yaxis=dict(range=[0, 1.15], tickformat=".0%"),
        legend=dict(orientation="h", y=-0.2),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_model_size_latency(comparison: dict):
    """Graphique taille vs latence."""
    models  = list(comparison.keys())
    sizes   = [comparison[m].get("model_size_mb", 0) for m in models]
    times   = [comparison[m].get("training_time_minutes", 0) for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Taille (MB)",
        x=models, y=sizes,
        marker_color="#9C27B0",
        text=[f"{s:.0f} MB" for s in sizes],
        textposition="outside",
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        name="Durée entraînement (min)",
        x=models, y=times,
        mode="lines+markers+text",
        marker=dict(size=10, color="#FF9800"),
        text=[f"{t:.0f}m" for t in times],
        textposition="top center",
        yaxis="y2",
    ))

    fig.update_layout(
        title="Taille modèle & Durée entraînement",
        yaxis=dict(title="Taille (MB)"),
        yaxis2=dict(title="Durée (min)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.2),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def _show_training_curves():
    """Affiche les courbes d'apprentissage sauvegardées."""
    curve_paths = {
        "YOLO" : MODELS_DIR / "yolo_finetuned" / "train" / "results.png",
        "ViT"  : MODELS_DIR / "vit_finetuned"  / "training_curves_vit.png",
        "SAM"  : MODELS_DIR / "sam_finetuned"  / "training_curves_sam.png",
    }

    available = {k: v for k, v in curve_paths.items() if v.exists()}

    if not available:
        st.info("Les courbes seront disponibles après l'entraînement.")
        return

    cols = st.columns(len(available))
    for col, (name, path) in zip(cols, available.items()):
        with col:
            st.markdown(f"**{name}**")
            st.image(str(path), use_container_width=True)


def _show_model_analysis():
    """Tableau d'analyse des forces/faiblesses."""
    data = {
        "Modèle": ["YOLOv8", "ViT", "SAM"],
        "Forces": [
            "Rapide, localise les objets, adapté aux scènes complexes",
            "Très précis pour la classification, léger à l'inférence",
            "Masques précis au pixel, très flexible avec les prompts",
        ],
        "Faiblesses": [
            "Ne classe qu'une boîte, pas de masque précis",
            "Ne localise pas, nécessite un seul objet par image",
            "Lent, nécessite un prompt, pas conçu pour la classification",
        ],
        "Usage idéal": [
            "Détection multi-objets en temps réel",
            "Classification rapide d'images simples",
            "Segmentation précise pour l'édition ou la médecine",
        ],
    }
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# ONGLET À PROPOS
# ════════════════════════════════════════════════════════════════════════════

def tab_about():
    """Onglet d'informations sur le projet."""
    st.header("À propos du projet")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Objectif")
        st.markdown("""
        Ce projet compare **3 modèles de vision** après fine-tuning sur un dataset
        de chiens, chats et oiseaux :

        | Modèle | Tâche | Méthode |
        |--------|-------|---------|
        | YOLOv8 | Détection | Fine-tuning complet |
        | ViT    | Classification | Transfer learning |
        | SAM    | Segmentation | Fine-tuning décodeur |
        """)

        st.subheader("Technologies")
        st.markdown("""
        - **Python 3.10+**
        - **PyTorch** & **HuggingFace Transformers**
        - **Ultralytics** (YOLOv8)
        - **Streamlit** (interface)
        - **Plotly** (visualisations)
        """)

    with col2:
        st.subheader("Structure du projet")
        st.code("""
vision_project/
├── data/
│   ├── train/   images + labels YOLO
│   ├── val/     images + labels YOLO
│   ├── test/    images + labels YOLO
│   └── vit_crops/  images par classe
├── models/
│   ├── yolo_finetuned/  best.pt + métriques
│   ├── vit_finetuned/   poids HuggingFace
│   └── sam_finetuned/   décodeur fine-tuné
├── src/
│   ├── prepare_dataset.py
│   ├── train_yolo.py
│   ├── train_vit.py
│   ├── train_sam.py
│   └── utils/
│       ├── metrics.py
│       └── visualize.py
└── app.py
        """, language="")

        st.subheader("Lancer les entraînements")
        st.code("""
# 1. Préparer le dataset
python src/prepare_dataset.py

# 2. Entraîner les modèles
python src/train_yolo.py
python src/train_vit.py
python src/train_sam.py

# 3. Lancer l'application
streamlit run app.py
        """, language="bash")


# ════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

def main():
    show_sidebar()

    st.title("Comparaison de Modèles Vision : YOLO · ViT · SAM")
    st.caption("Dogs, Cats & Birds Dataset — OFPPT TS DIA-IA-102")

    # Onglets principaux
    tab1, tab2, tab3 = st.tabs(["Inférence", "Comparaison", "À propos"])

    with tab1:
        tab_inference()

    with tab2:
        tab_comparison()

    with tab3:
        tab_about()


if __name__ == "__main__":
    main()
