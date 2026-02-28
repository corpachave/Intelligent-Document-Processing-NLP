import pdfplumber
import pandas as pd

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

text = extract_text_from_pdf(r"E:\Learning\Zaamila Development\Projects\Fintech-Intelligent-Document-Processing-NLP\data\raw_pdfs\loan_agreement_11.pdf")

with open(r"E:\Learning\Zaamila Development\Projects\Fintech-Intelligent-Document-Processing-NLP\sample_out\loan_agreement_11.csv", "w", encoding="utf-8") as f:
    f.write(text)

print("Extraction complete.")

df = pd.read_csv(r"E:\Learning\Zaamila Development\Projects\Fintech-Intelligent-Document-Processing-NLP\sample_out\loan_agreement_11.csv", sep="|")
print(df)
