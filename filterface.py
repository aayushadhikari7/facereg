import cv2
import numpy as np
import urllib.request
import os

# --- Configuration ---
# Path to the Haar Cascade XML file for face detection
# If the file doesn't exist locally, it will be downloaded.
HAARCASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
HAARCASCADE_FILENAME = "haarcascade_frontalface_default.xml"

# Placeholder image URLs for new filters.
# In a real application, you'd replace these with local paths to high-quality images
# or ensure proper downloading and handling of image assets.
# Using placehold.co for demonstration purposes.
EMOJI_OVERLAY_URL = "https://placehold.co/200x200/FFD700/000000?text=Emoji" # Yellow emoji placeholder
ANIMAL_OVERLAY_URL = "https://placehold.co/200x200/8B4513/FFFFFF?text=Animal+Face" # Brown animal face placeholder
FULL_HEAD_MASK_URL = "https://placehold.co/300x400/9400D3/FFFFFF?text=Full+Head+Mask" # Purple full head mask placeholder

# Local filenames for downloaded overlays
EMOJI_FILENAME = "emoji_overlay.png"
ANIMAL_FILENAME = "animal_overlay.png"
FULL_HEAD_MASK_FILENAME = "full_head_mask.png"

# --- Helper Function for Image Download ---
def download_file_if_not_exists(url, filename):
    """Downloads a file from a URL if it does not already exist locally."""
    if not os.path.exists(filename):
        print(f"Downloading {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"Download of {filename} complete.")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            print("Please check your internet connection or download the file manually.")
            return False
    return True

# --- Filter Functions ---

def apply_cartoon_filter(frame_roi):
    """
    Applies a basic cartoon-like filter to the given image region.
    This involves bilateral filtering for smoothing and adaptive thresholding for edge detection.
    """
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur to reduce noise while preserving edges
    gray = cv2.medianBlur(gray, 5)
    
    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(gray, 255, 
                                  cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 9, 9)
    
    # Apply bilateral filter for strong edge-preserving smoothing (color quantization effect)
    color = cv2.bilateralFilter(frame_roi, 9, 250, 250)
    
    # Combine color and edges to create the cartoon effect
    # Ensure edges is 3 channels for bitwise_and operation with color
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon_frame = cv2.bitwise_and(color, edges_bgr)
    
    return cartoon_frame

def apply_blur_filter(frame_roi, ksize=(99, 99)):
    """
    Applies a Gaussian blur filter to the given image region.
    `ksize` is the kernel size (width, height) - larger values mean more blur.
    """
    return cv2.GaussianBlur(frame_roi, ksize, 0)

def apply_pixelate_filter(frame_roi, pixel_size=10):
    """
    Applies a pixelation filter to the given image region.
    `pixel_size` determines the size of each "pixel" block.
    """
    h, w = frame_roi.shape[:2]
    
    # Resize to a smaller image
    temp = cv2.resize(frame_roi, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    
    # Resize back to original size, making it blocky
    pixelated_frame = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return pixelated_frame

def apply_invert_filter(frame_roi):
    """
    Inverts the colors of the given image region.
    """
    return cv2.bitwise_not(frame_roi)

def apply_sepia_filter(frame_roi):
    """
    Applies a sepia tone filter to the given image region.
    """
    # Define sepia matrix (common values)
    sepia_matrix = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]]).T # Transpose for correct multiplication
    
    # Convert to float for matrix multiplication
    frame_float = frame_roi.astype(np.float32)
    
    # Apply the sepia matrix
    sepia_frame = cv2.transform(frame_float, sepia_matrix)
    
    # Clip values to 0-255 and convert back to uint8
    sepia_frame = np.clip(sepia_frame, 0, 255).astype(np.uint8)
    
    return sepia_frame

def apply_overlay_filter(frame_roi, overlay_image, alpha=1.0):
    """
    Overlays a given image onto the face region, resizing it to fit.
    If the overlay has an alpha channel, it will be used for blending.
    Otherwise, a simple copy or weighted blending is performed.
    """
    if overlay_image is None:
        return frame_roi # Return original if overlay failed to load

    h_roi, w_roi = frame_roi.shape[:2]
    h_overlay, w_overlay = overlay_image.shape[:2]

    # Resize overlay to fit the face ROI
    # Maintain aspect ratio while fitting within face_roi dimensions
    scale = min(w_roi / w_overlay, h_roi / h_overlay)
    resized_overlay = cv2.resize(overlay_image, (int(w_overlay * scale), int(h_overlay * scale)), interpolation=cv2.INTER_AREA)

    h_resized, w_resized = resized_overlay.shape[:2]

    # Calculate position to center the overlay on the face ROI
    x_offset = (w_roi - w_resized) // 2
    y_offset = (h_roi - h_resized) // 2

    # Create a blank canvas the size of the face ROI
    # For blending, ensure it's BGR if overlay is BGR, or BGRA if overlay has alpha
    if resized_overlay.shape[2] == 4: # Has alpha channel
        # Create a BGRA canvas for blending
        temp_roi = np.zeros((h_roi, w_roi, 4), dtype=np.uint8)
        # Convert frame_roi to BGRA to blend with alpha channel
        frame_roi_bgra = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2BGRA)
        temp_roi[:,:,:3] = frame_roi_bgra[:,:,:3]
        temp_roi[:,:,3] = frame_roi_bgra[:,:,3]
    else: # No alpha channel, assume BGR
        temp_roi = frame_roi.copy()
    
    # Apply overlay
    for c in range(0, 3): # Iterate over BGR channels
        # Handle alpha channel if present in overlay
        if resized_overlay.shape[2] == 4:
            alpha_s = resized_overlay[:, :, 3] / 255.0 * alpha
            alpha_l = 1.0 - alpha_s

            for y in range(h_resized):
                for x in range(w_resized):
                    # Check bounds to prevent out-of-bounds access
                    if y_offset + y < h_roi and x_offset + x < w_roi:
                        temp_roi[y_offset + y, x_offset + x, c] = (
                            alpha_s[y, x] * resized_overlay[y, x, c] +
                            alpha_l[y, x] * temp_roi[y_offset + y, x_offset + x, c]
                        )
                        # Carry over original alpha for background if it was BGRA
                        if temp_roi.shape[2] == 4:
                            temp_roi[y_offset + y, x_offset + x, 3] = frame_roi_bgra[y_offset + y, x_offset + x, 3]
        else: # No alpha channel in overlay, simple blend
            # Ensure indices are within bounds
            y1, y2 = y_offset, y_offset + h_resized
            x1, x2 = x_offset, x_offset + w_resized
            if y1 < 0: y1 = 0
            if x1 < 0: x1 = 0
            if y2 > h_roi: y2 = h_roi
            if x2 > w_roi: x2 = w_roi

            if y2 > y1 and x2 > x1:
                # Blend using addWeighted for non-alpha images
                cv2.addWeighted(resized_overlay, alpha, temp_roi[y1:y2, x1:x2], 1 - alpha, 0, temp_roi[y1:y2, x1:x2])

    if temp_roi.shape[2] == 4:
        return cv2.cvtColor(temp_roi, cv2.COLOR_BGRA2BGR)
    return temp_roi


def apply_full_head_mask(frame_roi, full_head_mask_image):
    """
    Applies a larger mask (acting as a 'full head' replacement) over the face region.
    This function will attempt to cover the entire head area by making the overlay
    larger than the detected face ROI and positioning it slightly above.
    """
    if full_head_mask_image is None:
        return frame_roi

    h_roi, w_roi = frame_roi.shape[:2]

    # Calculate a size slightly larger than the face ROI to cover more of the head
    # Adjust scale factor as needed for your desired head coverage
    scale_factor = 1.5 # Make the mask 1.5 times larger than the detected face width
    
    # Calculate desired width and height for the mask, maintaining aspect ratio
    mask_w = int(w_roi * scale_factor)
    mask_h = int(full_head_mask_image.shape[0] * (mask_w / full_head_mask_image.shape[1]))

    resized_mask = cv2.resize(full_head_mask_image, (mask_w, mask_h), interpolation=cv2.INTER_AREA)

    # Calculate position to center the mask horizontally and place it slightly above the face
    # Adjust y_offset to move the mask upwards to cover the top of the head
    x_offset = (w_roi - mask_w) // 2
    y_offset = (h_roi - mask_h) // 2 - int(h_roi * 0.2) # Shift up by 20% of ROI height

    # Create a blank canvas for blending the resized mask
    # This ensures the mask can be placed partially outside the original face_roi if needed
    result_roi = frame_roi.copy()

    # Get dimensions for drawing
    y1_src, y2_src = 0, resized_mask.shape[0]
    x1_src, x2_src = 0, resized_mask.shape[1]

    y1_dest = max(0, y_offset)
    y2_dest = min(h_roi, y_offset + mask_h)
    x1_dest = max(0, x_offset)
    x2_dest = min(w_roi, x_offset + mask_w)

    # Adjust source crop if destination is out of bounds
    y1_src += (y1_dest - y_offset)
    y2_src += (y2_dest - (y_offset + mask_h))
    x1_src += (x1_dest - x_offset)
    x2_src += (x2_dest - (x_offset + mask_w))
    
    # Perform blending only if the source and destination regions are valid
    if y2_src > y1_src and x2_src > x1_src and y2_dest > y1_dest and x2_dest > x1_dest:
        if resized_mask.shape[2] == 4: # If mask has an alpha channel
            alpha_mask = resized_mask[y1_src:y2_src, x1_src:x2_src, 3] / 255.0
            for c in range(0, 3):
                result_roi[y1_dest:y2_dest, x1_dest:x2_dest, c] = (
                    alpha_mask * resized_mask[y1_src:y2_src, x1_src:x2_src, c] +
                    (1 - alpha_mask) * result_roi[y1_dest:y2_dest, x1_dest:x2_dest, c]
                )
        else: # No alpha channel, simple blending (e.g., for placehold.co images)
            cv2.addWeighted(resized_mask[y1_src:y2_src, x1_src:x2_src], 1.0, 
                            result_roi[y1_dest:y2_dest, x1_dest:x2_dest], 0.0, 0,
                            result_roi[y1_dest:y2_dest, x1_dest:x2_dest])

    return result_roi


# --- Main Application Logic ---

def main():
    """
    Main function to run the real-time face filter application.
    """
    # 1. Download Haar Cascade XML and overlay images if not present
    if not download_file_if_not_exists(HAARCASCADE_URL, HAARCASCADE_FILENAME):
        return
    if not download_file_if_not_exists(EMOJI_OVERLAY_URL, EMOJI_FILENAME):
        return
    if not download_file_if_not_exists(ANIMAL_OVERLAY_URL, ANIMAL_FILENAME):
        return
    if not download_file_if_not_exists(FULL_HEAD_MASK_URL, FULL_HEAD_MASK_FILENAME):
        return

    # 2. Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(HAARCASCADE_FILENAME)
    if face_cascade.empty():
        print(f"Error: Could not load face cascade classifier from {HAARCASCADE_FILENAME}.")
        print("Please check the file path and ensure it's not corrupted.")
        return

    # 3. Load overlay images
    # Using cv2.IMREAD_UNCHANGED to read potential alpha channel for PNGs
    emoji_img = cv2.imread(EMOJI_FILENAME, cv2.IMREAD_UNCHANGED)
    animal_img = cv2.imread(ANIMAL_FILENAME, cv2.IMREAD_UNCHANGED)
    full_head_img = cv2.imread(FULL_HEAD_MASK_FILENAME, cv2.IMREAD_UNCHANGED)

    if emoji_img is None: print(f"Warning: Could not load {EMOJI_FILENAME}. Emoji filter may not work.")
    if animal_img is None: print(f"Warning: Could not load {ANIMAL_FILENAME}. Animal face filter may not work.")
    if full_head_img is None: print(f"Warning: Could not load {FULL_HEAD_MASK_FILENAME}. Full head mask filter may not work.")

    # 4. Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Define available filters as a list for cycling
    filters_list = [
        {"name": "No Filter", "function": None},
        {"name": "Cartoon", "function": apply_cartoon_filter},
        {"name": "Blur", "function": apply_blur_filter},
        {"name": "Pixelation", "function": apply_pixelate_filter},
        {"name": "Invert Colors", "function": apply_invert_filter},
        {"name": "Sepia Tone", "function": apply_sepia_filter},
        {"name": "Emoji Overlay", "function": lambda roi: apply_overlay_filter(roi, emoji_img)},
        {"name": "Animal Face Overlay", "function": lambda roi: apply_overlay_filter(roi, animal_img)},
        {"name": "Full Head Mask", "function": lambda roi: apply_full_head_mask(roi, full_head_img)}
    ]
    current_filter_index = 0 # Start with no filter, now using index

    print("\n--- Face Filter Controls ---")
    print("Press 'F' to cycle through filters.")
    print("Press 'Q' to quit.")
    print("----------------------------\n")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Flip the frame horizontally for a mirror-like effect (optional)
        frame = cv2.flip(frame, 1)

        # Convert frame to grayscale for face detection (Haar cascades work on grayscale)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        # Parameters: image, scaleFactor, minNeighbors, minSize
        # scaleFactor: How much the image size is reduced at each image scale.
        # minNeighbors: How many neighbors each candidate rectangle should have to retain it.
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        # Apply filter to each detected face
        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) for the face
            face_roi = frame[y:y+h, x:x+w]

            # Apply the current filter if selected and ROI is valid
            selected_filter = filters_list[current_filter_index]
            if selected_filter["function"]:
                if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                    processed_face_roi = selected_filter["function"](face_roi)
                    # Replace the original face region with the processed one
                    frame[y:y+h, x:x+w] = processed_face_roi
            else:
                # If no filter, just draw a rectangle around the face (optional)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Blue rectangle

        # Display the current filter name on the frame
        current_filter_name = filters_list[current_filter_index]["name"]
        filter_name_text = f"Filter: {current_filter_name}"
        cv2.putText(frame, filter_name_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Face Filter App', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Press 'q' to quit
            break
        elif key == ord('f'): # Press 'f' to cycle through filters
            current_filter_index = (current_filter_index + 1) % len(filters_list)
            print(f"Switched to filter: {filters_list[current_filter_index]['name']}")

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
