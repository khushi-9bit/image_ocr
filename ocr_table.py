import os
import cv2
import numpy as np
import torch
from pdf2image import convert_from_path
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image

def load_model():
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
    return processor, model

def resize_image(image, max_size=1024):
    width, height = image.size
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
    return image

def detect_tables(image, processor, model, output_folder, page_num):
    image = Image.fromarray(image)
    image = resize_image(image)
    encoding = processor(image, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**encoding)
    
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    
    if 1 in predictions:  # 1 corresponds to 'table' class
        cv2.imwrite(os.path.join(output_folder, f'table_page_{page_num}.png'), np.array(image))

def process_pdf(pdf_path, output_folder):
    images = convert_from_path(pdf_path)
    os.makedirs(output_folder, exist_ok=True)
    processor, model = load_model()
    
    for i, image in enumerate(images):
        image_np = np.array(image)
        detect_tables(image_np, processor, model, output_folder, i)

if __name__ == "__main__":
    pdf_path = r"C:\Users\KhushiOjha\Downloads\ilovepdf_split(1)\geo_pdf-1-1.pdf"  # Change this to your PDF file path
    output_folder = "output_tablesssss"
    process_pdf(pdf_path, output_folder)
    print("Table detection complete. Check the output_tables folder.")

# import cv2
# import numpy as np
# import pytesseract
# from pdf2image import convert_from_path
# import os

# def preprocess_image(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
#     return binary

# def detect_tables(image, original_image, output_folder, page_num):
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
#     horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
#     vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
#     table_mask = cv2.add(horizontal_lines, vertical_lines)
#     contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     table_count = 0
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         if w > 50 and h > 50:
#             table_crop = original_image[y:y+h, x:x+w]
#             cv2.imwrite(os.path.join(output_folder, f'table_page_{page_num}_{table_count}.png'), table_crop)
#             table_count += 1

# def process_pdf(pdf_path, output_folder):
#     images = convert_from_path(pdf_path)
#     os.makedirs(output_folder, exist_ok=True)
    
#     for i, image in enumerate(images):
#         image_np = np.array(image)
#         processed_image = preprocess_image(image_np)
#         detect_tables(processed_image, image_np, output_folder, i)

# if __name__ == "__main__":
#     pdf_path = r"C:\Users\KhushiOjha\Downloads\ilovepdf_split(1)\geo_pdf-1-4.pdf" # Change this to your PDF file path
#     output_folder = "output_table"
#     process_pdf(pdf_path, output_folder)
#     print("Table detection complete. Check the output_tables folder.")
