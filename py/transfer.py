import os
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

src_folder = "F:/signal/code/heic/"
dst_folder = "F:/signal/code/dataset/map/images/train"
os.makedirs(dst_folder, exist_ok=True)

for filename in os.listdir(src_folder):
    if filename.lower().endswith(".heic"):
        heic_path = os.path.join(src_folder, filename)
        img = Image.open(heic_path)
        jpg_path = os.path.join(dst_folder, os.path.splitext(filename)[0] + ".jpg")
        img.save(jpg_path, format="JPEG")
        print(f"{filename} 转换完成 → {jpg_path}")
