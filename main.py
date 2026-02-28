import pdfplumber

# Detect PDF Type
def is_scanned_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        first_page = pdf.pages[0]
        text = first_page.extract_text()
        return text is None

# Digital PDF Extraction
def extract_digital_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text = text + page_text + "\n"
    return text

# OCR Pipeline
## Convert pdf to images
from pdf2image import convert_from_path

pages = convert_from_path(r"E:\Learning\Zaamila Development\Projects\Fintech-Intelligent-Document-Processing-NLP\data\raw_pdfs\scanned_pdf_samples\Image_pdf_merged_5pages.pdf", dpi=300)

## Image Preprocessing (Noise Reduction)
import cv2
import numpy as np

def preprocess_image(pil_image):
    img = np.array(pil_image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Noise removal
    gray = cv2.medianBlur(gray, 3)
    
    # Thresholding (binarization)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    return thresh
