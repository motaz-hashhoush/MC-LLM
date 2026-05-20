import os
import zipfile
import urllib.request
import sys

def progress_callback(block_num, block_size, total_size):
    if total_size > 0:
        downloaded = block_num * block_size
        progress = min(100, (downloaded / total_size) * 100)
        sys.stdout.write(f"\rDownloading: {progress:.1f}%")
        sys.stdout.flush()

def main():
    # Target directory on host
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "models", "tashkeel")
    os.makedirs(target_dir, exist_ok=True)
    
    zip_path = os.path.join(target_dir, "eo_model.zip")
    model_url = "https://huggingface.co/alif-mashreq/eo_model/resolve/main/eo_model.zip"
    
    # Download
    if not os.path.exists(zip_path):
        print(f"Downloading {model_url}...")
        try:
            # Add a User-Agent header to avoid 401/403 errors
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            
            urllib.request.urlretrieve(model_url, zip_path, progress_callback)
            print("\nDownload complete.")
        except Exception as e:
            print(f"\n❌ Error downloading: {e}")
            print("Tip: If you still get 401, you may need a HuggingFace token, but usually a User-Agent is enough for public files.")
            return
    else:
        print(f"File already exists: {zip_path}")
    
    # Extract
    print(f"Extracting to {target_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print("✅ Extraction complete.")
    except Exception as e:
        print(f"❌ Error extracting: {e}")
        return
    
    print("\n✅ eo_model ready and located at models/tashkeel/")

if __name__ == "__main__":
    main()
