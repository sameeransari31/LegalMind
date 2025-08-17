import os
import fitz
import docx
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup
from email import policy
from email.parser import BytesParser
from pdf2image import convert_from_path

from utils.helpers import clean_text

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_pdf(file_path):
    text = ""
    doc = fitz.open(file_path)

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        text += page_text

    # If text is very little, likely scanned — fallback to OCR
    if len(text.strip()) < 100:
        print("PDF might be image-based — using OCR")
        images = convert_from_path(file_path)
        for img_num, image in enumerate(images, start=1):
            ocr_text = pytesseract.image_to_string(image)
            text += ocr_text

    return clean_text(text)


def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
    except Exception as e:
        raise ValueError(f"Error reading DOCX file: {e}")

    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return clean_text("\n".join(full_text))


def extract_text_from_eml(file_path):
    with open(file_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                body += part.get_content()
            elif content_type == "text/html":
                soup = BeautifulSoup(part.get_content(), "html.parser")
                body += soup.get_text()
    else:
        body = msg.get_body(preferencelist=('plain', 'html')).get_content()

    return clean_text(body)


def extract_text(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        print("Detected PDF file.")
        return extract_text_from_pdf(file_path)

    elif ext == ".docx":
        print("Detected DOCX file.")
        return extract_text_from_docx(file_path)

    elif ext == ".eml":
        print("Detected EML email file.")
        return extract_text_from_eml(file_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
def load_document(file_path):
    return extract_text(file_path)
