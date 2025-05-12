import json
import os

def filter_annotations(image_folder_path, annotations_file_path, output_file_path):
    """
    Filters annotations in a JSON file based on the presence of corresponding images in a folder.

    Args:
        image_folder_path (str): The path to the folder containing the images.
        annotations_file_path (str): The path to the JSON annotations file.
        output_file_path (str): The path where the filtered JSON annotations will be saved.
    """
    try:
        # 1. Get the set of available image IDs from the image folder
        available_image_ids = set()
        if not os.path.isdir(image_folder_path):
            print(f"Error: Image folder not found at {image_folder_path}")
            return

        for filename in os.listdir(image_folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # Extract image_id (filename without extension)
                image_id = os.path.splitext(filename)[0]
                available_image_ids.add(image_id)

        if not available_image_ids:
            print(f"No image files found in {image_folder_path}. Cannot filter.")
            return

        print(f"Found {len(available_image_ids)} images in the folder.")

        # 2. Load the annotations JSON file
        try:
            with open(annotations_file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Annotations file not found at {annotations_file_path}")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {annotations_file_path}")
            return

        if 'annotations' not in data or not isinstance(data['annotations'], list):
            print(f"Error: The JSON file {annotations_file_path} does not have the expected structure (missing 'annotations' list).")
            return

        # 3. Filter the annotations
        original_count = len(data['annotations'])
        filtered_annotations = []
        removed_count = 0

        for annotation in data['annotations']:
            if 'image_id' in annotation and annotation['image_id'] in available_image_ids:
                filtered_annotations.append(annotation)
            else:
                if 'image_id' in annotation:
                    print(f"Removing annotation for image_id '{annotation['image_id']}' as image file is missing.")
                else:
                    print(f"Removing annotation due to missing 'image_id' field: {annotation}")
                removed_count += 1

        # 4. Create the new JSON structure for the output
        filtered_data = {"annotations": filtered_annotations}

        # 5. Save the filtered annotations to the output file
        try:
            with open(output_file_path, 'w') as f:
                json.dump(filtered_data, f, indent=4)
            print(f"\nSuccessfully filtered annotations.")
            print(f"Original number of annotations: {original_count}")
            print(f"Number of annotations removed: {removed_count}")
            print(f"Number of annotations remaining: {len(filtered_annotations)}")
            print(f"Filtered annotations saved to: {output_file_path}")
        except IOError:
            print(f"Error: Could not write the output file to {output_file_path}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Replace these paths with the actual paths to your files and folder.
    image_folder = "/home/cc/XrayGPT/dataset/openi/image"  # e.g., "/mnt/data/dataset_images" or "C:/Users/YourName/Desktop/images"
    json_annotation_file = "/home/cc/XrayGPT/dataset/openi/filter_cap.json"      # e.g., "/mnt/data/annotations.json"
    output_json_file = "/home/cc/XrayGPT/dataset/openi/filtered_annotations.json"
    # --- End Configuration ---

    # Ensure the user updates the paths before running
    if image_folder == "./path/to/your/image_folder" or json_annotation_file == "filter_cap.json":
        print("--------------------------------------------------------------------------")
        print("IMPORTANT: Please update the 'image_folder' and 'json_annotation_file' ")
        print("           variables in the script with your actual paths before running.")
        print("--------------------------------------------------------------------------")
    else:
        filter_annotations(image_folder, json_annotation_file, output_json_file)
