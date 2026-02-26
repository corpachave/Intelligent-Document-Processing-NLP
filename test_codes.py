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

pdf_path = r"E:\Learning\Zaamila Development\Projects\Fintech-Intelligent-Document-Processing-NLP\data\raw_pdfs\loan_agreement_01.pdf"

if is_scanned_pdf(pdf_path):
    print("This is a scanned pdf (Image based).")
else:
    print("This is a digital pdf (Text based).")

extracted_text = extract_digital_text(pdf_path)
print(extracted_text)