# Comparaison de Modèles Vision : YOLO · ViT · SAM

Application Streamlit de fine-tuning et comparaison de 3 modèles de vision par ordinateur sur un dataset chiens/chats/oiseaux.

**OFPPT — TS DIA-IA-102**

**Application déployée :** https://vision-models-comparison-abdelahfid-ait-lahcen.streamlit.app  
**Modèles fine-tunés :** https://huggingface.co/aitlahcenhafid/vision-models-ofppt  
**Code source :** https://github.com/aitlahcenhafid67-droid/vision-models-comparison

---

## Résultats obtenus

| Modèle | Tâche | Précision | Rappel | F1 | Métrique clé | Taille |
|--------|-------|-----------|--------|----|-------------|--------|
| YOLOv8n | Détection | 73.0% | 81.8% | 77.1% | mAP50 = **84.1%** | 6 MB |
| ViT-Base-Patch16 | Classification | 95.4% | 96.5% | 95.9% | Accuracy = **96.7%** | 327 MB |
| SAM ViT-Base | Segmentation | 100% | 100% | 100% | IoU = **1.000** | 15.5 MB |

---

## Structure du projet

```
vision_project/
├── app.py                    # Application Streamlit principale
├── requirements.txt          # Dépendances Python
├── src/
│   ├── prepare_dataset.py    # Extraction et préparation du dataset
│   ├── train_yolo.py         # Fine-tuning YOLOv8
│   ├── train_vit.py          # Fine-tuning ViT (head-only)
│   ├── train_sam.py          # Fine-tuning SAM (decoder-only)
│   └── utils/
│       ├── metrics.py        # Calcul des métriques
│       └── visualize.py      # Fonctions de visualisation
```

---

## Installation locale

### 1. Cloner le dépôt

```bash
git clone https://github.com/VOTRE_USERNAME/vision-models-comparison.git
cd vision-models-comparison
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Télécharger le dataset

Télécharger **Dogs, Cats and Birds** depuis Roboflow au format YOLOv8 et placer le ZIP dans le dossier `Downloads/`.

### 4. Préparer et entraîner

```bash
python src/prepare_dataset.py   # Prépare les données
python src/train_yolo.py        # Fine-tune YOLO (~20 époques)
python src/train_vit.py         # Fine-tune ViT (~10 époques)
python src/train_sam.py         # Fine-tune SAM (~5 époques)
```

> Les modèles de base (~700 MB au total) sont téléchargés automatiquement depuis HuggingFace lors du premier lancement.

### 5. Lancer l'application

```bash
streamlit run app.py
```

Ouvrir **http://localhost:8501**

---

## Description des modèles

### YOLOv8 — Détection d'objets
- Détecte et localise les objets avec des boîtes englobantes
- Fine-tuning complet (toutes les couches), 20 époques
- Métrique principale : **mAP50**

### ViT (Vision Transformer) — Classification
- Classifie l'image entière en une seule classe
- Stratégie **head-only** : encodeur gelé, seul le classifieur est entraîné (2 307 paramètres sur 86M)
- Métrique principale : **Accuracy**

### SAM (Segment Anything Model) — Segmentation
- Génère des masques pixel par pixel à partir d'une boîte de prompt
- Stratégie **decoder-only** : encodeur gelé, seul le décodeur est entraîné (4M sur 94M paramètres)
- Métrique principale : **IoU**

---

## Architecture de déploiement

```
GitHub (code)          HuggingFace Hub (modèles)     Streamlit Cloud (app)
──────────────         ─────────────────────────     ─────────────────────
app.py           +     yolo/best.pt (6 MB)      ───► App accessible en ligne
src/             +     vit/model.safetensors          télécharge les modèles
requirements.txt +     (327 MB)                       au premier démarrage
README.md              sam/sam_decoder.pth
                        (15.5 MB)
```

Les modèles fine-tunés sont hébergés sur HuggingFace Hub car trop lourds pour GitHub (limite 100 MB).  
L'application les télécharge automatiquement au premier démarrage.

---

## Technologies

- **Python 3.10+**
- **PyTorch** — framework deep learning
- **Ultralytics** — YOLOv8
- **HuggingFace Transformers** — ViT et SAM
- **HuggingFace Hub** — hébergement des modèles fine-tunés
- **Streamlit** — interface web
- **Plotly** — graphiques interactifs

---

*OFPPT — Filière Intelligence Artificielle — Niveau Technicien Spécialisé*  
*Formateur : Mr. SABER*
