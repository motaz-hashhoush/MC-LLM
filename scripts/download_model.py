"""
Model downloader script — downloads weights from HuggingFace to the local models directory.
"""

import os
import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download LLM models from HuggingFace.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen3-14B-AWQ",
        help="HuggingFace model ID (default: Qwen/Qwen3-14B-AWQ)"
    )
    parser.add_argument(
        "--dir", 
        type=str, 
        default="models",
        help="Local directory to store the model (default: models)"
    )
    
    args = parser.parse_args()
    
    # Resolve local directory
    # If running from project root, use models/MODEL_NAME
    # We strip the org name from the model ID for the directory name to keep it clean
    model_name_only = args.model.split("/")[-1]
    local_dir = os.path.join(args.dir, args.model)
    
    print(f"📥 Downloading '{args.model}' to '{local_dir}'...")
    
    try:
        snapshot_download(
            repo_id=args.model,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            revision="main"
        )
        print(f"✅ Download complete! Model is ready at: {local_dir}")
        print("\nTo use this with Docker Compose, ensure your .env has:")
        print(f"MODEL_NAME={args.model}")
    except Exception as e:
        print(f"❌ Error downloading model: {e}")

if __name__ == "__main__":
    # Ensure we are in the project root or similar
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    main()
