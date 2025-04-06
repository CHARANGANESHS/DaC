import os
import shutil
from pathlib import Path

def organize_breakhis_images(target_dir, target_to_image_dir_100x, target_to_image_dir_200x):
    os.makedirs(os.path.join(target_to_image_dir_100x, "benign"), exist_ok=True)
    os.makedirs(os.path.join(target_to_image_dir_100x, "malignant"), exist_ok=True)
    os.makedirs(os.path.join(target_to_image_dir_200x, "benign"), exist_ok=True)
    os.makedirs(os.path.join(target_to_image_dir_200x, "malignant"), exist_ok=True)
    
    children_directory_1 = ["benign", "malignant"]
    children_directory_2 = ["SOB"]
    children_directory_5 = ["100X", "200X"]
    
    print(f"Source directory: {target_dir}")
    print(f"Available top-level directories: {os.listdir(target_dir)}")
    
    total_images = 0
    processed_images = 0
    
    for d1 in children_directory_1:
        for d2 in children_directory_2:
            d2_path = os.path.join(target_dir, d1, d2) # ! "BreakHis_v1/benign/SOB/" or "BreakHis_v1/malignant/SOB/"
            if not os.path.exists(d2_path):
                print(f"Directory does not exist: {d2_path}")
                continue
                
            print(f"Processing directory: {d2_path}")
                
            for d3 in os.listdir(d2_path): # ! [adenosis, fibroadenoma, ...]
                d3_path = os.path.join(d2_path, d3) # ! "BreakHis_v1/benign/SOB/adenosis"
                if not os.path.isdir(d3_path):
                    continue
                    
                for d4 in os.listdir(d3_path): # ! [SOB, SOB, SOB]
                    d4_path = os.path.join(d3_path, d4)
                    if not os.path.isdir(d4_path):
                        continue
                    
                    for d5 in children_directory_5:
                        path = os.path.join(d4_path, d5)
                        if os.path.exists(path):
                            total_images += len([f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Found {total_images} images to process")
    malignant_count = 0
    benign_count = 0
    

    for d1 in children_directory_1:
        print(f"Processing {d1} images...")
        for d2 in children_directory_2:
            d2_path = os.path.join(target_dir, d1, d2)
            if not os.path.exists(d2_path):
                print(f"Directory does not exist: {d2_path}")
                continue
                
            for d3 in os.listdir(d2_path):
                d3_path = os.path.join(d2_path, d3)
                if not os.path.isdir(d3_path):
                    continue
                    
                for d4 in os.listdir(d3_path):
                    d4_path = os.path.join(d3_path, d4)
                    if not os.path.isdir(d4_path):
                        continue
                    
                    for d5 in children_directory_5:
                        source_path = os.path.join(d4_path, d5)
                        if not os.path.exists(source_path):
                            continue
                        
                        target_base = target_to_image_dir_100x if d5 == "100X" else target_to_image_dir_200x
                        
                        target_subdir = "benign" if d1 == "benign" else "malignant"
                        
                        target_dir_path = os.path.join(target_base, target_subdir)
                        
                        img_count = 0
                        
                        for img in os.listdir(source_path):
                            if img.endswith(('.png', '.jpg', '.jpeg')):
                                source_file = os.path.join(source_path, img)
                                target_file = os.path.join(target_dir_path, img)
                                
                                shutil.copy2(source_file, target_file)
                                processed_images += 1
                                img_count += 1
                                
                                if d1 == "malignant":
                                    malignant_count += 1
                                else:
                                    benign_count += 1
                                
                                if processed_images % 100 == 0:
                                    print(f"Processed {processed_images}/{total_images} images")
                        
                        if img_count > 0:
                            print(f"Copied {img_count} images from {source_path} to {target_dir_path}")
    
    print(f"\nCompleted! Processed {processed_images} images")
    print(f"Benign images: {benign_count}, Malignant images: {malignant_count}")
    print(f"100x/benign images stored in: {os.path.join(target_to_image_dir_100x, 'benign')}")
    print(f"100x/malignant images stored in: {os.path.join(target_to_image_dir_100x, 'malignant')}")
    print(f"200x/benign images stored in: {os.path.join(target_to_image_dir_200x, 'benign')}")
    print(f"200x/malignant images stored in: {os.path.join(target_to_image_dir_200x, 'malignant')}")

if __name__ == "__main__":
    target_dir = "data/BreaKHis_v1"
    target_to_image_dir_100x = "data/BreakHis/100x"
    target_to_image_dir_200x = "data/BreakHis/200x"
    
    organize_breakhis_images(target_dir, target_to_image_dir_100x, target_to_image_dir_200x) 