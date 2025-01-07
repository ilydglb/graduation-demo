import cv2
import numpy as np
import os

def simulate_radar_image(input_image_path, output_image_path, quality=90, compression=3):
    """
    Simulates a radar-like image from an RGB aerial photo.

    Parameters:
        input_image_path (str): Path to the input RGB image.
        output_image_path (str): Path to save the radar-like image.
        quality (int): JPEG quality (1-100, higher means better quality, larger file).
        compression (int): PNG compression (0-9, higher means smaller file, slower save).
    """
    # Step 1: Load the image
    rgb_image = cv2.imread(input_image_path)
    if rgb_image is None:
        raise FileNotFoundError(f"Image not found: {input_image_path}")

    # Convert BGR (OpenCV format) to RGB
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # Step 2: Convert to Grayscale
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # Step 3: Add Speckle Noise (Multiplicative Noise)
    noise = np.random.normal(0, 0.1, gray_image.shape)
    speckled_image = gray_image + gray_image * noise
    speckled_image = np.clip(speckled_image, 0, 255).astype(np.uint8)

    # Step 4: Apply Gaussian Blur to Simulate Radar Backscatter
    radar_blurred = cv2.GaussianBlur(speckled_image, (5, 5), 0)

    # Step 5: Edge Enhancement (Canny Edge Detection)
    edges = cv2.Canny(radar_blurred, threshold1=50, threshold2=150)
    radar_with_edges = cv2.addWeighted(radar_blurred, 0.8, edges, 0.2, 0)

    # Step 6: Adjust Dynamic Range
    radar_image = cv2.normalize(radar_with_edges, None, 0, 255, cv2.NORM_MINMAX)
    radar_image = radar_image.astype(np.uint8)

    # Step 7: Save with Compression Parameters
    file_extension = os.path.splitext(output_image_path)[1].lower()
    if file_extension == ".jpg" or file_extension == ".jpeg":
        cv2.imwrite(output_image_path, radar_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif file_extension == ".png":
        cv2.imwrite(output_image_path, radar_image, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    else:
        cv2.imwrite(output_image_path, radar_image)

    print(f"Radar-like image saved to: {output_image_path}")



def batch_process_images(input_folder, output_folder):
    """
    Batch processes all images in a folder to generate radar-like images.

    Parameters:
        input_folder (str): Folder containing input RGB images.
        output_folder (str): Folder to save radar-like images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, filename)

        try:
            simulate_radar_image(input_image_path, output_image_path)
        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")


# Example Usage
if __name__ == "__main__":
    input_folder = r"C:\Users\yilma\Desktop\graduation dataset\3\train\img" # Folder containing RGB images
    output_folder = r"C:\Users\yilma\Desktop\graduation dataset\3\train_radar\img"  # Folder to save radar images

    batch_process_images(input_folder, output_folder)
