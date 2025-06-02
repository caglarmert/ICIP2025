import os
import json
import gc
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import shape, Polygon, MultiPolygon
import pyvips
import shutil



# --- CONFIGURATION ---
NAME2LABEL = {
    "Others": 0,
    "T-G1": 1,
    "T-G2": 2,
    "T-G3": 3,
    "Normal mucosa": 4
}
LABEL2COLOR = {
    0: (0, 0, 0),
    1: (0, 192, 0),
    2: (255, 224, 32),
    3: (255, 0, 0),
    4: (0, 32, 255)
}
TILE_SIZE = 2048

def geojson_to_mask_tile(mask_img, geojson_path, scale, left, top, tile_width, img_width, img_height):
    """Draw annotations onto a downscaled tile of the mask"""
    with open(geojson_path) as f:
        data = json.load(f)

    draw = ImageDraw.Draw(mask_img)
    for feature in data['features']:
        geom = shape(feature['geometry'])
        props = feature['properties']
        classification = props.get('classification', {})
        name = classification.get('name', 'Others')
        label = NAME2LABEL.get(name, 0)
        color = LABEL2COLOR.get(label, (0, 0, 0))

        if isinstance(geom, Polygon):
            # Scale and translate coordinates
            coords = [
                (
                    int((x * scale) - (left * scale)),
                    int((y * scale) - (top * scale))
                )
                for x, y in geom.exterior.coords
            ]
            draw.polygon(coords, fill=color)
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                coords = [
                    (
                        int((x * scale) - (left * scale)),
                        int((y * scale) - (top * scale))
                    )
                    for x, y in poly.exterior.coords
                ]
                draw.polygon(coords, fill=color)


def convert_and_save_single_image(image_path, geojson_path, output_dir, tile_size=2048, scale_factor=0.1):
    name = os.path.splitext(os.path.basename(image_path))[0]
    img_save_path = os.path.join(output_dir, 'images', f"{name}.png")
    mask_save_path = os.path.join(output_dir, 'masks', f"{name}.png")
    # --- CHECK IF OUTPUT FILES ALREADY EXIST ---
    if os.path.exists(img_save_path) and os.path.exists(mask_save_path):
        print(f"[SKIP] Image and mask already exist for {name}, skipping conversion.")
        return

    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

    print(f"[START] Processing {name}...")

    # Open WSI
    img_vips = pyvips.Image.new_from_file(image_path, access='sequential')
    width, height = img_vips.width, img_vips.height
    print(f" - Original size: {width}x{height}")

    # Compute downsampled dimensions
    d_width = int(round(width * scale_factor))
    d_height = int(round(height * scale_factor))
    print(f" - Downsampled size: {d_width}x{d_height}")

    # Create NumPy arrays for resized image and mask
    resized_image_np = np.zeros((d_height, d_width, 3), dtype=np.uint8)
    resized_mask_np = np.zeros((d_height, d_width, 3), dtype=np.uint8)

    # Process image in tiles
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            tile_width = min(tile_size, width - left)
            tile_height = min(tile_size, height - top)

            # Crop and read tile
            region = img_vips.crop(left, top, tile_width, tile_height)
            region_np = np.ndarray(
                buffer=region.write_to_memory(),
                dtype=np.uint8,
                shape=[tile_height, tile_width, region.bands]
            )
            if region_np.shape[2] > 3:
                region_np = region_np[:, :, :3]

            # Calculate downscaled dimensions and positions
            d_l = int(round(left * scale_factor))
            d_t = int(round(top * scale_factor))
            d_l_next = int(round((left + tile_width) * scale_factor))
            d_t_next = int(round((top + tile_height) * scale_factor))
            d_tw = d_l_next - d_l
            d_th = d_t_next - d_t

            # Ensure we don't go out of bounds
            d_tw = int(min(d_tw, d_width - d_l))
            d_th = int(min(d_th, d_height - d_t))
            
            if d_tw <= 0 or d_th <= 0:
                continue

            # Resize tile to exactly match the calculated dimensions
            tile_img = Image.fromarray(region_np)
            resized_tile = tile_img.resize((d_tw, d_th), Image.BILINEAR)
            resized_tile_np = np.array(resized_tile)

            # Paste into final image
            resized_image_np[d_t:d_t + d_th, d_l:d_l + d_tw, :] = resized_tile_np

            # Draw mask for this tile
            mask_tile = Image.new("RGB", (d_tw, d_th), (0, 0, 0))
            geojson_to_mask_tile(mask_tile, geojson_path, scale_factor, left, top, tile_width, width, height)
            resized_mask_np[d_t:d_t + d_th, d_l:d_l + d_tw, :] = np.array(mask_tile)

            del region_np, resized_tile, mask_tile, resized_tile_np
            gc.collect()

    # Convert to VIPS and save
    vips_img = pyvips.Image.new_from_memory(
        resized_image_np.tobytes(), d_width, d_height, 3, format="uchar"
    )
    vips_mask = pyvips.Image.new_from_memory(
        resized_mask_np.tobytes(), d_width, d_height, 3, format="uchar"
    )

    vips_img.pngsave(img_save_path, compression=1)
    vips_mask.pngsave(mask_save_path, compression=1)

    print(f"[DONE] Saved downscaled image and mask for {name}")

    # Clean up
    del resized_image_np, resized_mask_np
    del vips_img, vips_mask
    gc.collect()


def batch_process_all(input_image_dir, geojson_dir, output_dir, tile_size=2048, scale_factor=0.1):
    image_files = [
        fname for fname in os.listdir(input_image_dir)
        if fname.lower().endswith(('.svs', '.tif', '.tiff'))
    ]

    for fname in image_files:
        img_path = os.path.join(input_image_dir, fname)
        name = os.path.splitext(fname)[0]
        geo_path = os.path.join(geojson_dir, f"{name}.geojson")

        if not os.path.exists(geo_path):
            print(f"[SKIP] GeoJSON not found for {fname}")
            continue

        print(f"\n--- STARTING PROCESS FOR: {fname} ---\n")
        try:
            convert_and_save_single_image(
                img_path, geo_path, output_dir, tile_size=tile_size, scale_factor=scale_factor
            )
        finally:
            gc.collect()
            print(f"\n--- FINISHED AND CLEANED UP FOR: {fname} ---\n")

    print("Finished all")


def test_convert_and_save_single_image(image_path, geojson_path, output_dir, tile_size=2048, scale_factor=0.1):
    name = os.path.splitext(os.path.basename(image_path))[0]
    img_save_path = os.path.join(output_dir, 'images', f"{name}.png")
    # --- CHECK IF OUTPUT FILES ALREADY EXIST ---
    if os.path.exists(img_save_path):
        print(f"[SKIP] Image already exist for {name}, skipping conversion.")
        return

    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    print(f"[START] Processing {name}...")

    # Open WSI
    img_vips = pyvips.Image.new_from_file(image_path, access='sequential')
    width, height = img_vips.width, img_vips.height
    print(f" - Original size: {width}x{height}")

    # Compute downsampled dimensions
    d_width = int(round(width * scale_factor))
    d_height = int(round(height * scale_factor))
    print(f" - Downsampled size: {d_width}x{d_height}")

    # Create NumPy arrays for resized image
    resized_image_np = np.zeros((d_height, d_width, 3), dtype=np.uint8)

    # Process image in tiles
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            tile_width = min(tile_size, width - left)
            tile_height = min(tile_size, height - top)

            # Crop and read tile
            region = img_vips.crop(left, top, tile_width, tile_height)
            region_np = np.ndarray(
                buffer=region.write_to_memory(),
                dtype=np.uint8,
                shape=[tile_height, tile_width, region.bands]
            )
            if region_np.shape[2] > 3:
                region_np = region_np[:, :, :3]

            # Calculate downscaled dimensions and positions
            d_l = int(round(left * scale_factor))
            d_t = int(round(top * scale_factor))
            d_l_next = int(round((left + tile_width) * scale_factor))
            d_t_next = int(round((top + tile_height) * scale_factor))
            d_tw = d_l_next - d_l
            d_th = d_t_next - d_t

            # Ensure we don't go out of bounds
            d_tw = int(min(d_tw, d_width - d_l))
            d_th = int(min(d_th, d_height - d_t))
            
            if d_tw <= 0 or d_th <= 0:
                continue

            # Resize tile to exactly match the calculated dimensions
            tile_img = Image.fromarray(region_np)
            resized_tile = tile_img.resize((d_tw, d_th), Image.BILINEAR)
            resized_tile_np = np.array(resized_tile)

            # Paste into final image
            resized_image_np[d_t:d_t + d_th, d_l:d_l + d_tw, :] = resized_tile_np

            del region_np, resized_tile, resized_tile_np
            gc.collect()

    # Convert to VIPS and save
    vips_img = pyvips.Image.new_from_memory(
        resized_image_np.tobytes(), d_width, d_height, 3, format="uchar"
    )

    vips_img.pngsave(img_save_path, compression=1)

    print(f"[DONE] Saved downscaled image for {name}")

    # Clean up
    del resized_image_np
    del vips_img
    gc.collect()

def test_batch_process_all(input_image_dir, geojson_dir, output_dir, tile_size=2048, scale_factor=0.1):
    image_files = [
        fname for fname in os.listdir(input_image_dir)
        if fname.lower().endswith(('.svs', '.tif', '.tiff'))
    ]

    for fname in image_files:
        img_path = os.path.join(input_image_dir, fname)
        name = os.path.splitext(fname)[0]
        geo_path = os.path.join(geojson_dir, f"{name}.geojson")

        if os.path.exists(geo_path):
            print(f"[SKIP] GeoJSON found for {fname}")
            continue

        print(f"\n--- STARTING PROCESS FOR: {fname} ---\n")
        try:
            test_convert_and_save_single_image(
                img_path, geo_path, output_dir, tile_size=tile_size, scale_factor=scale_factor
            )
        finally:
            gc.collect()
            print(f"\n--- FINISHED AND CLEANED UP FOR: {fname} ---\n")

    print("Finished all")

print("downscale by 20")
batch_process_all("images", "geojson", "20converted_output", TILE_SIZE, 0.05)
print("test 20")
test_batch_process_all("images", "geojson", "20test_output", TILE_SIZE, 0.05)
print("downscale by 40")
batch_process_all("images", "geojson", "40converted_output", TILE_SIZE, 0.025)
print("test 40")
test_batch_process_all("images", "geojson", "40test_output", TILE_SIZE, 0.025)


# Define paths
train_folder = "ICIP2025/dataset/annotations/train"
validation_folder = "ICIP2025/dataset/annotations/validation"
mixed_repository = "ICIP2025/svs/20converted_output/masks"
train_10_folder = "20/annotations/train"
validation_10_folder = "20/annotations/validation"

# Ensure target folders exist
os.makedirs(train_10_folder, exist_ok=True)
os.makedirs(validation_10_folder, exist_ok=True)

# Function to get all file names (excluding directories)
def get_file_list(folder):
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

# Get lists
train_files = get_file_list(train_folder)
validation_files = get_file_list(validation_folder)

# Copy matching files from mixed repository to new folders
for file_name in train_files:
    source_file = os.path.join(mixed_repository, file_name)
    dest_file = os.path.join(train_10_folder, file_name)
    if os.path.exists(source_file):
        shutil.copy(source_file, dest_file)

for file_name in validation_files:
    source_file = os.path.join(mixed_repository, file_name)
    dest_file = os.path.join(validation_10_folder, file_name)
    if os.path.exists(source_file):
        shutil.copy(source_file, dest_file)

print("Files copied successfully.")

train_folder = "ICIP2025/dataset/images/train"
validation_folder = "ICIP2025/dataset/images/validation"
mixed_repository = "ICIP2025/svs/20converted_output/images"
train_10_folder = "20/images/train"
validation_10_folder = "20/images/validation"

# Ensure target folders exist
os.makedirs(train_10_folder, exist_ok=True)
os.makedirs(validation_10_folder, exist_ok=True)

# Function to get all file names (excluding directories)
def get_file_list(folder):
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

# Get lists
train_files = get_file_list(train_folder)
validation_files = get_file_list(validation_folder)

# Copy matching files from mixed repository to new folders
for file_name in train_files:
    source_file = os.path.join(mixed_repository, file_name)
    dest_file = os.path.join(train_10_folder, file_name)
    if os.path.exists(source_file):
        shutil.copy(source_file, dest_file)

for file_name in validation_files:
    source_file = os.path.join(mixed_repository, file_name)
    dest_file = os.path.join(validation_10_folder, file_name)
    if os.path.exists(source_file):
        shutil.copy(source_file, dest_file)

print("Files copied successfully.")

source_folder = "ICIP2025/svs/20test_output/images"
destination_folder = "ICIP2025/svs/20/images/test"

# Ensure destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Copy contents
for item in os.listdir(source_folder):
    source_path = os.path.join(source_folder, item)
    dest_path = os.path.join(destination_folder, item)
    
    if os.path.isdir(source_path):
        shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
    else:
        shutil.copy2(source_path, dest_path)

print("Folder contents copied successfully.")

################
# Define paths
train_folder = "ICIP2025/dataset/annotations/train"
validation_folder = "ICIP2025/dataset/annotations/validation"
mixed_repository = "ICIP2025/svs/40converted_output/masks"
train_10_folder = "40/annotations/train"
validation_10_folder = "40/annotations/validation"

# Ensure target folders exist
os.makedirs(train_10_folder, exist_ok=True)
os.makedirs(validation_10_folder, exist_ok=True)

# Function to get all file names (excluding directories)
def get_file_list(folder):
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

# Get lists
train_files = get_file_list(train_folder)
validation_files = get_file_list(validation_folder)

# Copy matching files from mixed repository to new folders
for file_name in train_files:
    source_file = os.path.join(mixed_repository, file_name)
    dest_file = os.path.join(train_10_folder, file_name)
    if os.path.exists(source_file):
        shutil.copy(source_file, dest_file)

for file_name in validation_files:
    source_file = os.path.join(mixed_repository, file_name)
    dest_file = os.path.join(validation_10_folder, file_name)
    if os.path.exists(source_file):
        shutil.copy(source_file, dest_file)

print("Files copied successfully.")

train_folder = "ICIP2025/dataset/images/train"
validation_folder = "ICIP2025/dataset/images/validation"
mixed_repository = "ICIP2025/svs/40converted_output/images"
train_10_folder = "40/images/train"
validation_10_folder = "40/images/validation"

# Ensure target folders exist
os.makedirs(train_10_folder, exist_ok=True)
os.makedirs(validation_10_folder, exist_ok=True)

# Function to get all file names (excluding directories)
def get_file_list(folder):
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

# Get lists
train_files = get_file_list(train_folder)
validation_files = get_file_list(validation_folder)

# Copy matching files from mixed repository to new folders
for file_name in train_files:
    source_file = os.path.join(mixed_repository, file_name)
    dest_file = os.path.join(train_10_folder, file_name)
    if os.path.exists(source_file):
        shutil.copy(source_file, dest_file)

for file_name in validation_files:
    source_file = os.path.join(mixed_repository, file_name)
    dest_file = os.path.join(validation_10_folder, file_name)
    if os.path.exists(source_file):
        shutil.copy(source_file, dest_file)

print("Files copied successfully.")


source_folder = "ICIP2025/svs/40test_output/images"
destination_folder = "ICIP2025/svs/40/images/test"

# Ensure destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Copy contents
for item in os.listdir(source_folder):
    source_path = os.path.join(source_folder, item)
    dest_path = os.path.join(destination_folder, item)
    
    if os.path.isdir(source_path):
        shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
    else:
        shutil.copy2(source_path, dest_path)

print("Folder contents copied successfully.")
