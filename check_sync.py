import cv2
import numpy as np

# Read left and right images
parent_dir = '../camera_data/camera2/video/stopwatch/'  # Update this to your images directory

start_time=5.36
fps=30

for frame_id in range(1,150,10):  # Assuming you have 9 frames named left_001.png to left_009.png

    current_time = start_time + (frame_id-1) / fps

    left_image_path = parent_dir + f'left_{frame_id:03d}.png'
    print(f"Loading frame: {left_image_path}")
    right_image_path = parent_dir + f'right_{frame_id:03d}.png'
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)

    # Check if images are loaded
    if left_image is None or right_image is None:
        print("Error: Could not load images")
    else:
        # Resize images to same height if different
        height = max(left_image.shape[0], right_image.shape[0])
        left_image = cv2.resize(left_image, (int(left_image.shape[1] * height / left_image.shape[0]), height))
        right_image = cv2.resize(right_image, (int(right_image.shape[1] * height / right_image.shape[0]), height))
        
        # Concatenate images horizontally
        combined_image = np.hstack([left_image, right_image])
        
        # Display the combined image
        window_name = f'Frame {frame_id}, Time: {current_time:.2f}s'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 2000, 1000)
        cv2.imshow(window_name, combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()