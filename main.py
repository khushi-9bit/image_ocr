import cv2
import pytesseract
import numpy as np
# Load image
image = cv2.imread(r"ninebit_logo.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Apply adaptive thresholding (better for low contrast text)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Apply dilation and erosion to make text more clear 
kernel = np.ones((2,2), np.uint8)
processed = cv2.dilate(thresh, kernel, iterations=1)
processed = cv2.erode(processed, kernel, iterations=1)

# Extract text using OCR
text = pytesseract.image_to_string(processed, config="--psm 6")  

print("Extracted Text:", text)

""" import cv2
import pytesseract

# Load image
image = cv2.imread(r"ninebit_logo.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Extract text
text = pytesseract.image_to_string(thresh)

print("Extracted Text:", text)

"""