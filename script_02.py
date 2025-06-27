# RegNo: EG/2020/3845
# Take Home Assignment 2 - Computer Vision
# Q2

import cv2
import numpy as np
import os

# Create Results folder if it doesn't exist
os.makedirs("Results", exist_ok=True)

# Global list to store intermediate frames
video_frames = []

def collect_segmentation_frame(mask):
    # Convert mask to BGR for video
    frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    video_frames.append(frame)

def region_growing(image, seed_points, threshold_range):
    mask = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=np.uint8)
    image = image.astype(np.int32)

    step_counter = 0

    for seed_point in seed_points:
        seed_x, seed_y = seed_point
        if not (0 <= seed_x < image.shape[1] and 0 <= seed_y < image.shape[0]):
            continue

        seed_value = image[seed_y, seed_x]
        queue = [(seed_x, seed_y)]

        while queue:
            x, y = queue.pop(0)
            if visited[y, x]:
                continue

            visited[y, x] = 1
            if abs(image[y, x] - seed_value) <= threshold_range:
                mask[y, x] = 255

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                            if not visited[ny, nx]:
                                queue.append((nx, ny))

                # Save progress every 1000 pixels
                step_counter += 1
                if step_counter % 1000 == 0:
                    collect_segmentation_frame(mask.copy())

    return mask

# Load the image
image = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Could not load 'input.png'. Please check the path.")

# Define seed points and threshold
seed_points = [(490, 200), (680, 170), (400, 400)]
threshold_range = 10

# Run segmentation
segmented_image = region_growing(image, seed_points, threshold_range)

# Save final result image
cv2.imwrite("Results/final_segmented_image.png", segmented_image)

# Add final frame to video
collect_segmentation_frame(segmented_image.copy())

# Create video from stored frames
video_path = "Results/segmentation_process.mp4"
frame_height, frame_width = video_frames[0].shape[:2]
fps = 10  # Change this to control speed

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

for frame in video_frames:
    out.write(frame)
out.release()

# Display final image
cv2.imshow('Final Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
