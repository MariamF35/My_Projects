import os
from PIL import Image

def clean_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img.verify()
            except Exception:
                print(f"Removing corrupted file: {path}")
                os.remove(path)

if __name__ == "__main__":
    clean_folder("dataset")
