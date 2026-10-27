
import os
import io
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pypdf import PdfReader
from docx import Document

app = FastAPI(title="DocuAI API")

# Temporary development configuration.
# Restrict this to your actual CodePen origin before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def extract_text(filename: str, content: bytes) -> str:
    name = filename.lower()

    try:
        if name.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)

        if name.endswith(".docx"):
            doc = Document(io.BytesIO(content))
            paragraphs = [p.text for p in doc.paragraphs]
            for table in doc.tables:
                for row in table.rows:
                    paragraphs.append(" | ".join(cell.text for cell in row.cells))
            return "\n".join(paragraphs)

        if name.endswith(".txt"):
            return content.decode("utf-8-sig", errors="replace")

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read this document: {exc}",
        )

    raise HTTPException(
        status_code=400,
        detail="Unsupported file. Upload a PDF, DOCX, or TXT file.",
    )





def ask_gemini(prompt: str) -> str:
    import logging
    import time

    if not client:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key is not configured on the server.",
        )

    fallback_model = os.getenv(
        "GEMINI_FALLBACK_MODEL",
        "gemini-3.6-flash",
    )

    models = [MODEL_NAME]
    if fallback_model and fallback_model not in models:
        models.append(fallback_model)

    last_error = None

    for model in models:
        try:
            # Limit each request to 25 seconds.
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"http_options": {"timeout": 25000}},
            )

            if not response.text:
                raise HTTPException(
                    status_code=502,
                    detail="Gemini returned no text. Try another document.",
                )

            return response.text

        except HTTPException:
            raise

        except Exception as exc:
            last_error = exc
            code = getattr(exc, "code", None)

            logging.warning(
                "Gemini model %s failed with code %s: %s",
                model,
                code,
                type(exc).__name__,
            )

            # Try the fallback for temporary service errors.
            if code not in (429, 500, 503):
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"Gemini request failed with code {code}. "
                        "Check Render logs and API configuration."
                    ),
                )

            # Brief pause before attempting the fallback.
            time.sleep(1)

    logging.error("All Gemini models failed: %s", last_error)

    raise HTTPException(
        status_code=503,
        detail=(
            "Gemini is temporarily unavailable. "
            "Please try again in a minute."
        ),
    )




@app.get("/")
def home():
    return {"message": "DocuAI API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    filename = file.filename or ""
    content = await file.read()

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Maximum file size is 10 MB.")

    text = extract_text(filename, content).strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="No readable text found. Scanned PDFs need OCR, which is not included yet.",
        )

    if len(text) > 100_000:
        text = text[:100_000]

    summary = ask_gemini(
        "Summarize the following document clearly. Use a short overview, "
        "important points, and a conclusion. Do not invent facts.\n\n"
        f"DOCUMENT:\n{text}"
    )

    return {"filename": filename, "summary": summary}


@app.post("/chat")
async def chat(
    question: str = Form(...),
    document_text: str = Form(...),
):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Enter a question.")

    if len(document_text) > 100_000:
        document_text = document_text[:100_000]

    answer = ask_gemini(
        "Answer the user's question using the supplied document. "
        "If the answer is not in the document, say so clearly.\n\n"
        f"DOCUMENT:\n{document_text}\n\n"
        f"QUESTION:\n{question}"
    )

    return {"answer": answer}
