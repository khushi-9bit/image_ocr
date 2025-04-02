import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    extracted_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # If text is extracted, add it and skip OCR
                extracted_text += page_text + "\n"
                continue  # Skip OCR for this page

    images = convert_from_path(pdf_path, dpi=300)  # Convert pages to images
    for img in images:
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

        # Apply thresholding
        _, img_cv = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform OCR only if `pdfplumber` didn't extract text
        extracted_text += pytesseract.image_to_string(img_cv) + "\n"

    return extracted_text

# Example usage
pdf_path = r"C:\Users\KhushiOjha\Downloads\geo_chap_9.pdf"
text = extract_text_from_pdf(pdf_path)
print("Extracted Text:\n", text)

for page_num in range(len(doc)):
    page = doc[page_num]
    page_rect = page.rect  # Get page dimensions
    page_area = page_rect.width * page_rect.height  # Total page area

    image_list = page.get_images(full=True)
    total_image_area = 0

    for img in image_list:
        xref = img[0]
        bbox = page.get_image_rects(xref)  # Get image bounding box

        for rect in bbox:
            img_width = rect.width
            img_height = rect.height
            image_area = img_width * img_height
            total_image_area += image_area

    # Calculate percentage of space occupied by images
    image_percentage = (total_image_area / page_area) * 100 

    print(f"Page {page_num + 1}: {image_percentage:.2f}% of the page is occupied by images.") 





# import pdfplumber
# import pytesseract
# from pdf2image import convert_from_path
# import cv2
# import numpy as np
# import fitz  # PyMuPDF

# def extract_text_from_pdf(pdf_path):
#     extracted_text = ""

#     #Step 1: Extract direct text from the PDF
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             extracted_text += page.extract_text() + "\n"

#     #Step 2: Extract text from images in the PDF
#     images = convert_from_path(pdf_path, dpi=300)  # Convert PDF pages to images
#     for img in images:
#         # Convert PIL image to OpenCV format
#         img_cv = np.array(img)
#         img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

#         # Apply thresholding for better OCR accuracy
#         _, img_cv = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#         # Perform OCR
#         extracted_text += pytesseract.image_to_string(img_cv) + "\n"

#     return extracted_text

# # Example usage
# pdf_path = r"C:\Users\KhushiOjha\Downloads\image_space (3).pdf"
# text = extract_text_from_pdf(pdf_path)
# print("Extracted Text:\n", text)
# doc = fitz.open(pdf_path)

# for page_num in range(len(doc)):
#     page = doc[page_num]
#     page_rect = page.rect  # Get page dimensions
#     page_area = page_rect.width * page_rect.height  # Total page area

#     image_list = page.get_images(full=True)
#     total_image_area = 0
#     unique_images = set()  # Track unique images to avoid duplicates

#     for img in image_list:
#         xref = img[0]

#         # Check if this image has already been counted
#         if xref in unique_images:
#             continue  # Skip duplicate images
#         unique_images.add(xref)  # Mark this image as processed

#         bbox = page.get_image_rects(xref)  # Get image bounding box

#         for rect in bbox:
#             img_width = rect.width
#             img_height = rect.height
#             image_area = img_width * img_height

#             # Ensure we don't exceed the page area (fixes large percentage issues)
#             if image_area > page_area:
#                 image_area = page_area  # Cap the area to page size

#             total_image_area += image_area

#     # Calculate percentage of space occupied by images
#     image_percentage = min((total_image_area / page_area) * 100, 100)  # Ensure it never exceeds 100%

#     print(f"Page {page_num + 1}: {image_percentage:.2f}% of the page is occupied by images.") 
    

