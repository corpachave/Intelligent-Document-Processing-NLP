import pdfplumber

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

text = extract_text_from_pdf(r"E:\Learning\Zaamila Development\Project details\Projects\Fintech---Intelligent-Document-Processing-NLP\data\raw_pdfs\sample_agreement.pdf")

with open(r"E:\Learning\Zaamila Development\Project details\Projects\Fintech---Intelligent-Document-Processing-NLP\data\raw_pdfs\sample_agreement.pdf", "w", encoding="utf-8") as f:
    f.write(text)

print("Extraction complete.")
