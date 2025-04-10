import pdfplumber
import pytesseract
import cv2
import numpy as np
import json
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os 

def detect_borders_opencv(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border_boxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 1000]
    return border_boxes

def is_text_inside_border(word_bbox, borders):
    x, y, w, h = word_bbox
    for bx, by, bw, bh in borders:
        if bx <= x and by <= y and (bx + bw) >= (x + w) and (by + bh) >= (y + h):
            return True
    return False

def extract_text_with_ocr(image, borders):
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    normal_text, bordered_text = [], []
    for i in range(len(ocr_data["text"])):
        if ocr_data["text"][i].strip():
            word_bbox = (
                ocr_data["left"][i], ocr_data["top"][i],
                ocr_data["width"][i], ocr_data["height"][i]
            )
            if is_text_inside_border(word_bbox, borders):
                bordered_text.append(ocr_data["text"][i])
            else:
                normal_text.append(ocr_data["text"][i])
    return " ".join(normal_text), " ".join(bordered_text)

def process_page(page_num, page, image):
    try:
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        borders = detect_borders_opencv(image_cv)
        normal_text, bordered_text = extract_text_with_ocr(image_cv, borders)

        tables = []
        pdf_tables = page.extract_tables()
        if pdf_tables:
            for table in pdf_tables:
                if table and len(table) > 1:
                    headers = table[0]
                    structured = []
                    for row in table[1:]:
                        row_dict = {
                            headers[i].strip(): row[i].strip()
                            for i in range(len(headers)) if headers[i] and row[i]
                        }
                        structured.append(row_dict)
                    if structured:
                        tables.append(structured)

        return {
            "page_number": page_num + 1,
            "normal_text": normal_text.strip(),
            "bordered_text": bordered_text.strip(),
            "tables": tables
        }
    except Exception as e:
        print(f"Error processing page {page_num + 1}: {e}")
        return {
            "page_number": page_num + 1,
            "normal_text": "",
            "bordered_text": "",
            "tables": [],
            "error": str(e)
        }

def extract_data_from_pdf(pdf_path, max_threads=12):
    extracted_data = []
    images = convert_from_path(pdf_path, dpi=300)

    with pdfplumber.open(pdf_path) as pdf:
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = []
            for page_num, (page, image) in enumerate(zip(pdf.pages, images)):
                futures.append(executor.submit(process_page, page_num, page, image))

            for future in as_completed(futures):
                extracted_data.append(future.result())

    # Sort to maintain page order
    return sorted(extracted_data, key=lambda x: x["page_number"])

if __name__ == "__main__":
    pdf_path = r"D:\History.pdf"
    start = time.time()
    print("🔍 Extracting data from PDF using multithreading...")
    output_data = extract_data_from_pdf(pdf_path, max_threads=12)  # You can increase threads here
    
    print("💾 Saving extracted data to output.json...")
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    end = time.time()
    print(f"duration: {(end - start):.4f}")
    print("✅ Done!")

