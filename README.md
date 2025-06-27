# EE7204 – Computer Vision and Image Processing  
## Take Home Assignment 2

This project contains Python programs to perform image segmentation and thresholding techniques as part of the EE7204 coursework.

### Tasks

1. **Add Gaussian Noise + Otsu's Thresholding**  
   - An image with 2 objects and 3 distinct gray levels (object1, object2, background) is used.  
   - Gaussian noise is added to simulate real-world image degradation.  
   - Otsu’s algorithm is implemented to automatically find the optimal threshold for separating objects from the background.

2. **Region-Growing Segmentation**  
   - A region-growing algorithm is implemented starting from predefined seed points.
   - Neighboring pixels are added to the region if their intensity falls within a user-defined threshold relative to the seed value.
   - The segmentation process is visualized and saved as a video.

---

### How to Use

1. Place the grayscale input image as `input.png` in the same folder as the scripts.
2. Run the Python scripts:
   - `script_01.py` – for noise addition and thresholding
   - `script_02.py` – for region-growing based segmentation
3. Output images and videos will be saved in the `Results` folder.

---

### Requirements

- Python 3  
- OpenCV (`cv2`)  
- NumPy (`numpy`)

---


### Author

**Bandara N.M.K.M.**  
Reg No: **EG/2020/3845**
