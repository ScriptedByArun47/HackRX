import requests
import mimetypes
import fitz  # PyMuPDF
import docx
import email
from email import policy
from bs4 import BeautifulSoup
from io import BytesIO
from app.logger import logger  # Uses your centralized logger

# --- File extractors ---

def extract_text_from_pdf(file_bytes):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        logger.error("PDF extraction failed.", exc_info=True)
        return ""

def extract_text_from_docx(file_bytes):
    try:
        doc = docx.Document(BytesIO(file_bytes))
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        logger.error("DOCX extraction failed.", exc_info=True)
        return ""

def extract_text_from_txt(file_bytes):
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.error("TXT decoding failed.", exc_info=True)
        return ""

def extract_text_from_eml(file_bytes):
    try:
        msg = email.message_from_bytes(file_bytes, policy=policy.default)
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    return part.get_content()
                elif content_type == "text/html":
                    html = part.get_content()
                    soup = BeautifulSoup(html, "html.parser")
                    return soup.get_text()
        else:
            return msg.get_content()
    except Exception as e:
        logger.error("EML parsing failed.", exc_info=True)
        return ""

# --- Clause splitter ---

def split_into_clauses(text: str, min_words=10, chunk_size=50):
    text = text.replace("\r", "").replace("\xa0", " ").strip()
    blocks = [b.strip() for b in text.split("\n\n") if len(b.strip()) > 40]

    if len(blocks) < 5:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        buffer = ""
        blocks = []
        for line in lines:
            buffer += " " + line
            if len(buffer.split()) >= chunk_size:
                blocks.append(buffer.strip())
                buffer = ""
        if buffer:
            blocks.append(buffer.strip())

    return [{"clause": block} for block in blocks if len(block.split()) >= min_words]

# --- Entry point ---

def extract_clauses_from_url(url: str):
    logger.info(f"Starting extraction from URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        file_bytes = response.content

        if not file_bytes:
            logger.warning("Empty file content received.")
            return []

        content_type = response.headers.get("Content-Type", "")
        mime_type, _ = mimetypes.guess_type(url)
        mime_type = (mime_type or content_type or "").lower()
        logger.info(f"Detected MIME type: {mime_type}")

        # Detect and extract raw text
        if "pdf" in mime_type:
            raw_text = extract_text_from_pdf(file_bytes)
        elif "docx" in mime_type:
            raw_text = extract_text_from_docx(file_bytes)
        elif "plain" in mime_type or url.lower().endswith(".txt"):
            raw_text = extract_text_from_txt(file_bytes)
        elif "rfc822" in mime_type or url.lower().endswith(".eml"):
            raw_text = extract_text_from_eml(file_bytes)
        else:
            logger.warning(f"Unknown MIME type '{mime_type}', defaulting to PDF parser.")
            raw_text = extract_text_from_pdf(file_bytes)

        if not raw_text.strip():
            logger.warning("No text extracted from the document.")
            return []

        return split_into_clauses(raw_text)

    except Exception as e:
        logger.error(f"Failed to extract clauses from: {url}", exc_info=True)
        return []
