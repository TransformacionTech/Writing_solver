# ============================================
# INGEST — Indexa los posts de post/ en ChromaDB
# ============================================
# Uso: python knowledge/ingest.py
# ============================================
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv(override=True)

# ── Rutas ─────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent          # raíz del proyecto
POSTS_DIR   = BASE_DIR / "post"                     # carpeta con los .docx
CHROMA_DIR  = Path(__file__).parent / "chroma_db"   # índice vectorial

COLLECTION  = "posts_tas"
EMBED_MODEL = "text-embedding-3-small"


def leer_docx(path: Path) -> str:
    """Extrae el texto de un archivo .docx."""
    from docx import Document
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def leer_archivo(path: Path) -> str:
    """Lee texto según la extensión del archivo."""
    ext = path.suffix.lower()
    if ext == ".docx":
        return leer_docx(path)
    elif ext in (".md", ".txt"):
        return path.read_text(encoding="utf-8")
    return ""


def indexar() -> int:
    """
    Lee todos los archivos de post/, genera embeddings con OpenAI
    y los almacena en ChromaDB. Devuelve el número de documentos indexados.
    """
    client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chroma  = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col     = chroma.get_or_create_collection(COLLECTION)

    archivos = list(POSTS_DIR.glob("*.docx")) + \
               list(POSTS_DIR.glob("*.md"))   + \
               list(POSTS_DIR.glob("*.txt"))

    if not archivos:
        print(f"[ingest] No se encontraron archivos en {POSTS_DIR}")
        return 0

    indexados = 0
    for archivo in archivos:
        texto = leer_archivo(archivo).strip()
        if not texto:
            print(f"[ingest] Saltando {archivo.name} (vacío)")
            continue

        respuesta = client.embeddings.create(
            input=texto,
            model=EMBED_MODEL
        )
        embedding = respuesta.data[0].embedding

        col.upsert(
            ids=[archivo.stem],
            documents=[texto],
            embeddings=[embedding],
            metadatas=[{"fuente": archivo.name}]
        )
        indexados += 1
        print(f"[ingest] Indexado: {archivo.name}")

    print(f"\n[ingest] Base actualizada: {indexados} documentos | collection='{COLLECTION}'")
    return indexados


if __name__ == "__main__":
    total = indexar()
    sys.exit(0 if total >= 0 else 1)
