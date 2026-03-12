from src.ocr.extractor import extract_text, is_scanned_pdf

# sample scanned pdf
pdf_file = r"data\raw_pdfs\scanned_pdf_samples\Image_pdf_01.pdf"
# sample digital pdf
#pdf_file = r"data\raw_pdfs\loan_agreement_01.pdf"

print("Checking PDF type...")
print("Scanned PDF:", is_scanned_pdf(pdf_file))

print("\nExtracted Text:\n")

text = extract_text(pdf_file)

print(text[:1000])   # print first 1000 characters