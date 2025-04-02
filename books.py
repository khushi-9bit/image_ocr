import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import json

def extract_tables_from_pdf(pdf_path):
    extracted_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_content = {"page_number": page_num + 1, "tables": []}
            
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    if table and len(table) > 1:
                        headers = table[0]  # Assume first row as headers
                        structured_table = []
                        for row in table[1:]:
                            row_dict = {headers[i]: row[i] for i in range(len(headers)) if row[i] is not None}
                            structured_table.append(row_dict)
                        page_content["tables"].append(structured_table)
            
            extracted_data.append(page_content)
    
    return extracted_data

def extract_text_from_images(pdf_path):
    extracted_data = []
    images = convert_from_path(pdf_path, dpi=300)
    
    for page_num, img in enumerate(images):
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        _, img_cv = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        extracted_text = pytesseract.image_to_string(img_cv).strip()
        extracted_data.append({"page_number": page_num + 1, "ocr_text": extracted_text})
    
    return extracted_data

def save_extracted_data(pdf_path, output_json):
    tables_data = extract_tables_from_pdf(pdf_path)
    
    if not any(page["tables"] for page in tables_data):  # If no tables, try OCR
        tables_data = extract_text_from_images(pdf_path)
    
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(tables_data, json_file, ensure_ascii=False, indent=4)
    
    print(f"✅ Extracted data saved to: {output_json}")

# Example usage
pdf_path = r"C:\Users\KhushiOjha\Downloads\geo_chap_9.pdf"
output_json = "extracted_data.json"
save_extracted_data(pdf_path, output_json)



# import pdfplumber
# import pytesseract
# from pdf2image import convert_from_path
# import cv2
# import numpy as np
# import json

# def process_pdf_page(page, page_num, text_data):
#     """
#     Extracts text and tables from a page and stores them in JSON format.
#     """
#     page_content = {"page_number": page_num + 1, "text": "", "tables": []}

#     # Step 1: Extract text
#     text = page.extract_text()
#     if text:
#         page_content["text"] = text.strip()

#     # Step 2: Extract tables properly
#     tables = page.extract_tables()
#     if tables:
#         for table in tables:
#             # Convert table list into structured dictionaries
#             if table and len(table) > 1:
#                 headers = table[0]  # First row as headers
#                 structured_table = []
#                 for row in table[1:]:
#                     row_dict = {headers[i]: row[i] for i in range(len(headers)) if row[i] is not None}
#                     structured_table.append(row_dict)
#                 page_content["tables"].append(structured_table)

#     text_data.append(page_content)  # Store structured page data

# def extract_text_from_images(pdf_path, page_num, text_data):
#     """
#     Converts a PDF page to an image and extracts text using OCR.
#     """
#     ocr_content = {"page_number": page_num + 1, "ocr_text": "", "tables": []}

#     images = convert_from_path(pdf_path, dpi=300, first_page=page_num + 1, last_page=page_num + 1)
#     for img in images:
#         img_cv = np.array(img)
#         img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

#         # Apply thresholding
#         _, img_cv = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#         # Perform OCR
#         extracted_text = pytesseract.image_to_string(img_cv).strip()
#         ocr_content["ocr_text"] = extracted_text

#     text_data.append(ocr_content)  # Store OCR text

# def process_pdf(pdf_path, output_json_path):
#     """
#     Extracts text and tables from a PDF and saves them in a JSON file.
#     """
#     text_data = []  # Stores structured text and tables data

#     with pdfplumber.open(pdf_path) as pdf:
#         for page_num, page in enumerate(pdf.pages):
#             process_pdf_page(page, page_num, text_data)

#             # If no text found, perform OCR
#             if not text_data[-1]["text"]:
#                 extract_text_from_images(pdf_path, page_num, text_data)

#     # Save text and table data as JSON
#     with open(output_json_path, "w", encoding="utf-8") as json_file:
#         json.dump(text_data, json_file, ensure_ascii=False, indent=4)

#     print(f"\n✅ Extracted data saved to: {output_json_path}")

# # Example Usage
# pdf_path = r"C:\Users\KhushiOjha\Downloads\ilovepdf_split(1)\geo_pdf-1-1.pdf"
# output_json_path = "extracted_data.json"

# # Process the PDF
# process_pdf(pdf_path, output_json_path)

# import pdfplumber
# import pytesseract
# from pdf2image import convert_from_path
# import cv2
# import numpy as np
# import fitz  # PyMuPDF

# def process_pdf_page(page, page_num):
#     """
#     Process each PDF page dynamically, extracting text, tables, and images in order.
#     """
#     extracted_content = f"\n---- Page {page_num + 1} ----\n"
#     text_found = False

#     # Step 1: Try extracting direct text
#     text = page.extract_text()
#     if text:
#         extracted_content += "[Extracted Text]\n" + text + "\n"
#         text_found = True

#     # Step 2: Try extracting tables
#     tables = page.extract_tables()
#     if tables:
#         extracted_content += "[Extracted Tables]\n"
#         for table in tables:
#             for row in table:
#                 extracted_content += " | ".join(row) + "\n"
#         text_found = True  # Consider tables as valid extracted content

#     return extracted_content, text_found

# def extract_text_from_images(pdf_path, page_num):
#     """
#     Converts a specific PDF page to an image and extracts text using OCR.
#     """
#     extracted_text = f"\n[OCR Extracted Text - Page {page_num + 1}]\n"
    
#     images = convert_from_path(pdf_path, dpi=300, first_page=page_num + 1, last_page=page_num + 1)
#     for img in images:
#         img_cv = np.array(img)
#         img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

#         # Apply thresholding
#         _, img_cv = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#         # Perform OCR
#         extracted_text += pytesseract.image_to_string(img_cv) + "\n"

#     return extracted_text

# def calculate_image_percentage(pdf_path):
#     """
#     Analyzes the amount of space occupied by images in each page.
#     """
#     doc = fitz.open(pdf_path)
#     image_percentages = []

#     for page_num in range(len(doc)):
#         page = doc[page_num]
#         page_rect = page.rect
#         page_area = page_rect.width * page_rect.height

#         image_list = page.get_images(full=True)
#         total_image_area = 0

#         for img in image_list:
#             xref = img[0]
#             bbox = page.get_image_rects(xref)

#             for rect in bbox:
#                 img_width, img_height = rect.width, rect.height
#                 image_area = min(img_width * img_height, page_area)  # Cap at page size
#                 total_image_area += image_area

#         image_percentage = min((total_image_area / page_area) * 100, 100)
#         image_percentages.append(f"Page {page_num + 1}: {image_percentage:.2f}% of the page is occupied by images.")

#     return image_percentages

# def process_pdf(pdf_path):
#     """
#     Iterates through the PDF pages and dynamically extracts text, tables, or images.
#     """
#     extracted_data = ""
    
#     with pdfplumber.open(pdf_path) as pdf:
#         for page_num, page in enumerate(pdf.pages):
#             page_content, text_found = process_pdf_page(page, page_num)
#             extracted_data += page_content
            
#             # If no text or tables found, try OCR on the images
#             if not text_found:
#                 extracted_data += extract_text_from_images(pdf_path, page_num)

#     # Analyze image usage on each page
#     image_info = calculate_image_percentage(pdf_path)
    
#     return extracted_data, image_info

# # Example Usage
# pdf_path = r"C:\Users\KhushiOjha\Downloads\ilovepdf_split(1)\geo_pdf-1-1.pdf"

# # Process PDF dynamically
# extracted_content, image_info = process_pdf(pdf_path)

# # Display extracted data
# print("Extracted Content:\n", extracted_content)
# print("\nImage Coverage Info:\n", "\n".join(image_info))


# import pdfplumber
# import pytesseract
# from pdf2image import convert_from_path
# import cv2
# import numpy as np
# import fitz  # PyMuPDF

# def extract_text_and_tables(pdf_path):
#     extracted_text = ""
#     extracted_tables = []

#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             # Extract Text
#             page_text = page.extract_text()
#             if page_text:
#                 extracted_text += page_text + "\n"

#             # Extract Tables
#             tables = page.extract_tables()
#             for table in tables:
#                 extracted_tables.append(table)

#     return extracted_text, extracted_tables

# def extract_text_from_images(pdf_path):
#     extracted_text = ""
#     images = convert_from_path(pdf_path, dpi=300)  # Convert PDF pages to images
    
#     for img in images:
#         img_cv = np.array(img)
#         img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        
#         # Apply thresholding for better OCR accuracy
#         _, img_cv = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#         # Perform OCR
#         extracted_text += pytesseract.image_to_string(img_cv) + "\n"

#     return extracted_text

# def calculate_image_percentage(pdf_path):
#     doc = fitz.open(pdf_path)
#     image_percentages = []

#     for page_num in range(len(doc)):
#         page = doc[page_num]
#         page_rect = page.rect
#         page_area = page_rect.width * page_rect.height

#         image_list = page.get_images(full=True)
#         total_image_area = 0
#         unique_images = set()  # Track unique images

#         for img in image_list:
#             xref = img[0]
#             if xref in unique_images:
#                 continue  # Avoid duplicate processing
#             unique_images.add(xref)

#             bbox = page.get_image_rects(xref)
#             for rect in bbox:
#                 img_width, img_height = rect.width, rect.height
#                 image_area = img_width * img_height
#                 total_image_area += min(image_area, page_area)  # Cap at page size

#         # Calculate image percentage
#         image_percentage = min((total_image_area / page_area) * 100, 100)
#         image_percentages.append(f"Page {page_num + 1}: {image_percentage:.2f}% of the page is occupied by images.")

#     return image_percentages

# # Example Usage
# pdf_path = r"C:\Users\KhushiOjha\Downloads\ilovepdf_split(1)\geo_pdf-1-1.pdf"

# # Extract text and tables
# text, tables = extract_text_and_tables(pdf_path)

# # Extract text from images (if needed)
# ocr_text = extract_text_from_images(pdf_path)

# # Calculate image coverage percentage
# image_info = calculate_image_percentage(pdf_path)

# # Display Results
# print("Extracted Text:\n", text)
# print("Extracted Text from Images:\n", ocr_text)
# print("Image Coverage Info:\n", "\n".join(image_info))

# # Display extracted tables
# if tables:
#     for i, table in enumerate(tables, start=1):
#         print(f"\nTable {i}:")
#         for row in table:
#             print(row)
