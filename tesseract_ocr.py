import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

## Run Tesseract OCR
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np

def preprocess_image(pil_image):
    # Convert PIL image to OpenCV format
    open_cv_image = np.array(pil_image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    return thresh

def ocr_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang="eng")
    return text

def extract_scanned_text(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300)
    full_text = ""
    
    for page in pages:
        processed = preprocess_image(page)
        text = pytesseract.image_to_string(processed, lang="eng")
        full_text += text + "\n"
    
    return full_text

result = ocr_image(r"E:\Learning\Zaamila Development\Projects\Fintech-Intelligent-Document-Processing-NLP\data\raw_pdfs\scanned_pdf_samples\scanned_image_jpg.jpg")
print("Result:\n",result)

pdf_text = extract_scanned_text(r"E:\Learning\Zaamila Development\Projects\Fintech-Intelligent-Document-Processing-NLP\data\raw_pdfs\loan_agreement_01.pdf")
print("PDF Text: \n",pdf_text)