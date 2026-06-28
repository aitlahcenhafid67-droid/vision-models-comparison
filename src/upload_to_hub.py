"""
Upload des modeles fine-tunes vers HuggingFace Hub.
Lancer une seule fois depuis le PC local.

Usage:
    python src/upload_to_hub.py --token TON_TOKEN_HF --username TON_USERNAME
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
REPO_NAME    = "vision-models-ofppt"   # nom du repo HuggingFace

def upload(token: str, username: str):
    api  = HfApi()
    repo = f"{username}/{REPO_NAME}"

    print(f"Creation du repo : {repo}")
    create_repo(repo, token=token, repo_type="model", exist_ok=True, private=False)

    files_to_upload = [
        (MODELS_DIR / "yolo_finetuned" / "train" / "weights" / "best.pt",  "yolo/best.pt"),
        (MODELS_DIR / "vit_finetuned"  / "model.safetensors",               "vit/model.safetensors"),
        (MODELS_DIR / "vit_finetuned"  / "config.json",                     "vit/config.json"),
        (MODELS_DIR / "sam_finetuned"  / "sam_decoder_finetuned.pth",       "sam/sam_decoder_finetuned.pth"),
    ]

    for local_path, hub_path in files_to_upload:
        if not local_path.exists():
            print(f"  MANQUANT : {local_path}")
            continue
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  Upload {hub_path} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=hub_path,
            repo_id=repo,
            token=token,
        )
        print(f"  OK : {hub_path}")

    print(f"\nModeles disponibles sur : https://huggingface.co/{repo}")
    print(f"Copie cette ligne dans app.py :")
    print(f'  HF_REPO = "{repo}"')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token",    required=True, help="HuggingFace access token")
    parser.add_argument("--username", required=True, help="Ton username HuggingFace")
    args = parser.parse_args()
    upload(args.token, args.username)
