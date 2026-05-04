"""
Model downloader script — downloads weights from HuggingFace to the local models directory.

Downloads:
  1. LLM model (Qwen/Qwen3-14B-AWQ or as specified)
  2. SILMA TTS model (silma-ai/silma-tts)
"""

import os
import sys
import argparse
from huggingface_hub import snapshot_download


def download_llm_model(model_name: str, local_dir: str) -> None:
    """Download an LLM model from HuggingFace Hub to local_dir."""
    print(f"📥 Downloading LLM '{model_name}' to '{local_dir}'...")
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            revision="main",
        )
        print(f"✅ LLM download complete! Model is ready at: {local_dir}")
        print("\nTo use this with Docker Compose, ensure your .env has:")
        print(f"MODEL_NAME={model_name}")
    except Exception as e:
        print(f"❌ Error downloading LLM model: {e}")
        sys.exit(1)


def download_silma_tts(model_name: str, local_dir: str) -> None:
    """
    Download silma-ai/silma-tts from HuggingFace Hub to local_dir.

    Skips the download if model files are already present (idempotent).

    Parameters
    ----------
    model_name:
        HuggingFace repository ID, e.g. ``"silma-ai/silma-tts"``.
    local_dir:
        Local directory where the model will be saved,
        e.g. ``"models/silma-ai/silma-tts"``.
    """
    # Consider already downloaded if the directory is non-empty
    if os.path.isdir(local_dir) and any(
        f for f in os.listdir(local_dir) if not f.startswith(".")
    ):
        print(f"✅ SILMA TTS already downloaded at '{local_dir}' — skipping.")
        return

    print(f"📥 Downloading SILMA TTS '{model_name}' to '{local_dir}'...")
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            revision="main",
        )
        print(f"✅ SILMA TTS download complete! Model is ready at: {local_dir}")
    except Exception as e:
        print(f"❌ Error downloading SILMA TTS model: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download models from HuggingFace.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-14B-AWQ",
        help="HuggingFace LLM model ID (default: Qwen/Qwen3-14B-AWQ)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="models",
        help="Local base directory to store models (default: models)",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Skip downloading the SILMA TTS model.",
    )
    parser.add_argument(
        "--tts-only",
        action="store_true",
        help="Download only the SILMA TTS model (skip LLM).",
    )

    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    # ── LLM ──────────────────────────────────────────────────────────────────
    if not args.tts_only:
        llm_local_dir = os.path.join(args.dir, args.model)
        download_llm_model(args.model, llm_local_dir)

    # ── SILMA TTS ─────────────────────────────────────────────────────────────
    if not args.skip_tts:
        # Read from environment / settings if available, else use defaults
        try:
            # Try loading from project settings for consistency
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from app.config import get_settings
            settings = get_settings()
            tts_model_name = settings.TTS_MODEL_NAME
            tts_local_dir = os.path.join(args.dir, settings.TTS_MODEL_NAME)
        except Exception:
            tts_model_name = "silma-ai/silma-tts"
            tts_local_dir = os.path.join(args.dir, tts_model_name)

        download_silma_tts(tts_model_name, tts_local_dir)


if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    main()
