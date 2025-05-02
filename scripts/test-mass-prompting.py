import csv
import time
import logging
import os
import glob
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
from PIL.PngImagePlugin import PngInfo

load_dotenv()

from openai import OpenAI
import csv
import time
import requests
from tqdm import tqdm
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("image_generation.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load prompt data from the CSV file
logger.info("Reading prompt data from prompts-sumi-v2.csv")
prompt_data = []
try:
    with open(
        "temp/prompts-sumi-v2.csv", mode="r", encoding="utf-8", newline=""
    ) as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip header row

        # Verify header structure with new title field
        expected_header = ["title", "prompt", "filename", "alttext", "description"]
        if len(header) >= 5 and all(
            h1 == h2 for h1, h2 in zip(header[:5], expected_header)
        ):
            for row in csv_reader:
                if len(row) >= 5:
                    prompt_data.append(
                        {
                            "title": row[0],
                            "prompt": row[1],
                            "filename": row[2],
                            "alttext": row[3],
                            "description": row[4],
                        }
                    )
                else:
                    logger.warning(f"Skipping row with insufficient columns: {row}")
        else:
            logger.error(
                f"CSV header doesn't match expected format. Expected: {','.join(expected_header)}, Got: {','.join(header)}"
            )
            raise ValueError("CSV header format mismatch")

    logger.info(f"Successfully loaded {len(prompt_data)} prompt entries from CSV file")
except Exception as e:
    logger.error(f"Error reading CSV file: {str(e)}")
    raise

# Create base output directory if it doesn't exist
base_output_dir = "temp/generated_images"
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)
    logger.info(f"Created base output directory: {base_output_dir}")


# Create a new batch folder
def create_batch_folder():
    # Find existing batch folders
    existing_batches = glob.glob(os.path.join(base_output_dir, "batch*"))

    # Determine the next batch number
    if not existing_batches:
        next_batch_num = 1
    else:
        # Extract batch numbers from folder names
        batch_nums = []
        for batch_folder in existing_batches:
            batch_name = os.path.basename(batch_folder)
            try:
                # Extract the number from batchXXX
                num = int(batch_name[5:])
                batch_nums.append(num)
            except ValueError:
                continue

        if batch_nums:
            next_batch_num = max(batch_nums) + 1
        else:
            next_batch_num = 1

    # Create the new batch folder
    batch_folder = os.path.join(base_output_dir, f"batch{next_batch_num:03d}")
    os.makedirs(batch_folder, exist_ok=True)
    logger.info(f"Created new batch folder: {batch_folder}")

    return batch_folder


# Create a new batch folder for this run
output_dir = create_batch_folder()


# Function to generate a single image from a prompt
def generate_single_image(prompt, idx, variant_num):
    data = prompt_data[idx]
    logger.info(
        f"Starting generation for image {idx + 1}/{len(prompt_data)} variant {variant_num} - {data['title']}"
    )
    start_time = time.time()

    # Make a single image request
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,  # DALL-E 3 only supports n=1
            size="1024x1024",  # You can change to "512x512" or "1792x1024" for widescreen
            quality="standard",  # Options: 'standard', 'hd'
            style="vivid",  # Options: 'vivid', 'natural'
        )

        end_time = time.time()
        generation_time = end_time - start_time
        logger.info(
            f"Image {idx + 1} variant {variant_num} generated in {generation_time:.2f} seconds"
        )

        return response.data[0].url
    except Exception as e:
        logger.error(
            f"Error generating image {idx + 1} variant {variant_num}: {str(e)}"
        )
        raise


# Function to generate 4 images by making 4 separate API calls
def generate_4_variants(prompt, idx):
    data = prompt_data[idx]
    logger.info(
        f"Starting generation of 4 variants for prompt {idx + 1}/{len(prompt_data)} - {data['title']}"
    )
    overall_start = time.time()

    # Generate 4 variants with separate API calls
    image_urls = []
    for variant_num in range(1, 5):
        try:
            # Add slight variation to prompt for diversity
            variant_prompt = prompt
            if variant_num > 1:
                variant_prompt = f"{prompt} Unique variation {variant_num}."

            url = generate_single_image(variant_prompt, idx, variant_num)
            image_urls.append(url)

            # Slight pause between API calls to be respectful
            if variant_num < 4:
                time.sleep(1)

        except Exception as e:
            logger.error(f"Failed to generate variant {variant_num}: {str(e)}")
            # Continue with other variants even if one fails

    overall_end = time.time()
    total_generation_time = overall_end - overall_start
    logger.info(
        f"Generated {len(image_urls)}/{4} variants in {total_generation_time:.2f} seconds"
    )

    return image_urls


# Function to download an image with progress tracking and add metadata
def download_image(url, data, idx, variant_num):
    logger.info(
        f"Starting download for image {idx + 1} variant {variant_num} - {data['title']}"
    )
    start_time = time.time()

    # Get file size
    response = requests.head(url)
    file_size = int(response.headers.get("content-length", 0))

    # Sanitize filename to ensure it's valid
    base_filename = os.path.basename(data["filename"])
    file_name, file_ext = os.path.splitext(base_filename)

    # If no extension, add .png
    if not file_ext or file_ext.lower() not in [".png", ".jpg", ".jpeg"]:
        file_ext = ".png"

    # Create variant filename
    variant_filename = f"{file_name}_v{variant_num}{file_ext}"
    file_path = os.path.join(output_dir, variant_filename)

    # Download with progress tracking
    logger.info(f"Downloading to {file_path} ({file_size/1024:.1f} KB)")
    response = requests.get(url, stream=True)

    # Use tqdm for progress bar
    temp_file_path = file_path + ".temp"
    with open(temp_file_path, "wb") as f, tqdm(
        desc=f"Image {idx + 1} variant {variant_num}",
        total=file_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

    # Add metadata to the image file
    try:
        # Create metadata for PNG
        metadata = PngInfo()
        metadata.add_text("Title", data["title"])
        metadata.add_text("Prompt", data["prompt"])
        metadata.add_text("Alt-Text", data["alttext"])
        metadata.add_text("Description", data["description"])
        metadata.add_text("Generated", datetime.now().isoformat())
        metadata.add_text("Batch", os.path.basename(output_dir))
        metadata.add_text("Variant", str(variant_num))

        # Open the image, add metadata, and save
        img = Image.open(temp_file_path)
        img.save(file_path, pnginfo=metadata)

        # Remove temporary file
        os.remove(temp_file_path)
        logger.info(f"Added metadata to {file_path}")
    except Exception as e:
        # If metadata addition fails, just rename the temp file
        logger.warning(f"Failed to add metadata: {str(e)}")
        os.rename(temp_file_path, file_path)

    end_time = time.time()
    download_time = end_time - start_time
    logger.info(
        f"Image {idx + 1} variant {variant_num} downloaded and processed in {download_time:.2f} seconds"
    )

    return file_path, file_size


# Start time for overall process
overall_start = time.time()
successful_generations = 0  # Number of prompts successfully processed
successful_downloads = 0  # Number of individual images downloaded
total_expected_images = len(prompt_data) * 4  # 4 images per prompt

# Write batch info to file
with open(
    os.path.join(output_dir, "_batch_info.txt"), "w", encoding="utf-8"
) as batch_info:
    batch_info.write(f"Batch: {os.path.basename(output_dir)}\n")
    batch_info.write(f"Date: {datetime.now().isoformat()}\n")
    batch_info.write(f"Number of prompts: {len(prompt_data)}\n")
    batch_info.write(f"Images per prompt: 4\n")
    batch_info.write(f"Total expected images: {total_expected_images}\n")
    batch_info.write("\n=== Prompt Data ===\n\n")
    for i, data in enumerate(prompt_data):
        batch_info.write(f"Item {i+1}:\n")
        batch_info.write(f"  Title: {data['title']}\n")
        batch_info.write(f"  Filename: {data['filename']}\n")
        batch_info.write(f"  Prompt: {data['prompt']}\n")
        batch_info.write(f"  Alt Text: {data['alttext']}\n")
        batch_info.write(f"  Description: {data['description']}\n\n")

# Loop through prompts and generate images
generated_files = []  # Track all generated files
for idx, data in enumerate(prompt_data):
    try:
        # Progress indicator
        progress_percent = (idx / len(prompt_data)) * 100
        logger.info(
            f"Processing entry {idx + 1}/{len(prompt_data)} ({progress_percent:.1f}%)"
        )
        logger.info(f"Title: {data['title']}")
        logger.info(f"Prompt summary: {data['prompt'][:50]}...")

        # Generate 4 images (via 4 separate API calls)
        image_urls = generate_4_variants(data["prompt"], idx)
        if image_urls:
            successful_generations += 1

        # Track the variant files for this prompt
        prompt_files = []

        # Download all variants
        for variant_num, url in enumerate(image_urls, 1):
            try:
                # Download the image and add metadata
                file_path, file_size = download_image(url, data, idx, variant_num)
                successful_downloads += 1
                prompt_files.append(
                    {"path": file_path, "size": file_size, "variant": variant_num}
                )

                # Log file details
                logger.info(
                    f"Saved image variant {variant_num} to {file_path} ({file_size/1024:.1f} KB)"
                )
            except Exception as e:
                logger.error(
                    f"Error processing variant {variant_num} for entry {idx + 1}: {str(e)}"
                )

        # Add all successful variants to our tracking
        generated_files.append(
            {"prompt_idx": idx, "title": data["title"], "files": prompt_files}
        )

    except Exception as e:
        logger.error(f"Error processing entry {idx + 1}: {str(e)}")
        # Continue with next prompt despite errors

# Calculate and log summary stats
overall_end = time.time()
total_time = overall_end - overall_start
average_time_per_prompt = total_time / len(prompt_data) if prompt_data else 0
average_time_per_image = (
    total_time / successful_downloads if successful_downloads else 0
)

# Update batch info file with results
with open(
    os.path.join(output_dir, "_batch_info.txt"), "a", encoding="utf-8"
) as batch_info:
    batch_info.write("\n=== Results ===\n\n")
    batch_info.write(
        f"Successful prompt generations: {successful_generations}/{len(prompt_data)}\n"
    )
    batch_info.write(
        f"Successful image downloads: {successful_downloads}/{total_expected_images}\n"
    )
    batch_info.write(f"Total execution time: {total_time:.2f} seconds\n")
    batch_info.write(
        f"Average time per prompt (4 images): {average_time_per_prompt:.2f} seconds\n"
    )
    batch_info.write(
        f"Average time per individual image: {average_time_per_image:.2f} seconds\n"
    )

# Create a CSV with results for easier future processing
with open(
    os.path.join(output_dir, "_batch_results.csv"), "w", encoding="utf-8", newline=""
) as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "title",
            "prompt",
            "base_filename",
            "alttext",
            "description",
            "variant",
            "output_filename",
            "status",
            "file_size_kb",
        ]
    )

    for idx, data in enumerate(prompt_data):
        # Find all variants for this prompt
        variants_found = [f for f in generated_files if f["prompt_idx"] == idx]

        if variants_found and len(variants_found[0]["files"]) > 0:
            # Write a row for each successfully generated variant
            for variant in variants_found[0]["files"]:
                filename = os.path.basename(variant["path"])
                csv_writer.writerow(
                    [
                        data["title"],
                        data["prompt"],
                        data["filename"],
                        data["alttext"],
                        data["description"],
                        variant["variant"],
                        filename,
                        "Success",
                        f"{variant['size']/1024:.1f}",
                    ]
                )
        else:
            # If no variants were successful, write a failure row
            for v in range(1, 5):
                csv_writer.writerow(
                    [
                        data["title"],
                        data["prompt"],
                        data["filename"],
                        data["alttext"],
                        data["description"],
                        v,
                        "",
                        "Failed",
                        "",
                    ]
                )

logger.info("=" * 50)
logger.info(f"Batch folder: {output_dir}")
logger.info("Generation Summary:")
logger.info(f"Total prompt entries processed: {len(prompt_data)}")
logger.info(f"Total images expected: {total_expected_images}")
logger.info(f"Successful prompt generations: {successful_generations}")
logger.info(f"Successful image downloads: {successful_downloads}")
logger.info(f"Total execution time: {total_time:.2f} seconds")
logger.info(
    f"Average time per prompt (4 images): {average_time_per_prompt:.2f} seconds"
)
logger.info(f"Average time per individual image: {average_time_per_image:.2f} seconds")
logger.info("=" * 50)
