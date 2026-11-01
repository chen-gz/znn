#!/usr/bin/env python3
"""
Convert Tiny ImageNet (extracted from tiny-imagenet-200.zip) into high-performance packed binary format.
Record format: <1 byte class_id (0..199)><12288 bytes pixel values (3 x 64 x 64, planar CHW: RRR...GGG...BBB...)>
Total per sample: 12,289 bytes.
"""

import os
import sys
from PIL import Image

def pack_split(base_dir, split_name, output_bin, class_to_id):
    split_dir = os.path.join(base_dir, split_name)
    if not os.path.exists(split_dir):
        print(f"Skipping {split_name}: {split_dir} does not exist.")
        return

    print(f"Packing {split_name} into {output_bin}...")
    sample_count = 0
    with open(output_bin, "wb") as f_out:
        if split_name == "train":
            # Structure: train/<class_name>/images/<image_file>.JPEG
            for cls_name in sorted(os.listdir(split_dir)):
                cls_path = os.path.join(split_dir, cls_name, "images")
                if not os.path.isdir(cls_path):
                    continue
                cid = class_to_id.get(cls_name)
                if cid is None:
                    continue
                cid_byte = bytes([cid])
                for fname in sorted(os.listdir(cls_path)):
                    if not fname.lower().endswith((".jpg", ".jpeg")):
                        continue
                    fpath = os.path.join(cls_path, fname)
                    with Image.open(fpath) as img:
                        img_rgb = img.convert("RGB")
                        r, g, b = img_rgb.split()
                        # Planar CHW
                        raw_bytes = cid_byte + r.tobytes() + g.tobytes() + b.tobytes()
                        f_out.write(raw_bytes)
                        sample_count += 1
        elif split_name == "val":
            # Structure: val/val_annotations.txt mapping image file to class name
            anno_path = os.path.join(split_dir, "val_annotations.txt")
            img_dir = os.path.join(split_dir, "images")
            if not os.path.exists(anno_path) or not os.path.exists(img_dir):
                print(f"Skipping val: missing annotations or images directory.")
                return
            with open(anno_path, "r") as f_anno:
                for line in f_anno:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    fname, cls_name = parts[0], parts[1]
                    cid = class_to_id.get(cls_name)
                    if cid is None:
                        continue
                    fpath = os.path.join(img_dir, fname)
                    if not os.path.exists(fpath):
                        continue
                    cid_byte = bytes([cid])
                    with Image.open(fpath) as img:
                        img_rgb = img.convert("RGB")
                        r, g, b = img_rgb.split()
                        raw_bytes = cid_byte + r.tobytes() + g.tobytes() + b.tobytes()
                        f_out.write(raw_bytes)
                        sample_count += 1

    print(f"Packed {sample_count} samples into {output_bin} ({os.path.getsize(output_bin)} bytes).")

def main():
    base_dir = "data/tiny-imagenet-200"
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]

    if not os.path.exists(base_dir):
        print(f"Error: directory '{base_dir}' not found.")
        print("Please download and unzip tiny-imagenet-200.zip first:")
        print("  zig build run-download-data -- tiny_imagenet")
        print("  unzip -q data/tiny-imagenet-200.zip -d data/")
        sys.exit(1)

    wnids_path = os.path.join(base_dir, "wnids.txt")
    if not os.path.exists(wnids_path):
        print(f"Error: wnids.txt not found in {base_dir}")
        sys.exit(1)

    class_to_id = {}
    with open(wnids_path, "r") as f:
        for idx, line in enumerate(f):
            class_to_id[line.strip()] = idx

    print(f"Loaded {len(class_to_id)} classes from {wnids_path}.")
    os.makedirs("data", exist_ok=True)
    pack_split(base_dir, "train", "data/tiny_imagenet_train.bin", class_to_id)
    pack_split(base_dir, "val", "data/tiny_imagenet_val.bin", class_to_id)
    print("Done! Tiny ImageNet binary datasets are ready.")

if __name__ == "__main__":
    main()
