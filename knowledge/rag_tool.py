# ============================================
# RAG TOOL — Herramienta de búsqueda semántica
# para agentes CrewAI sobre posts de Tech And Solve
# ============================================
import os
from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(override=True)

CHROMA_DIR  = Path(__file__).parent / "chroma_db"
COLLECTION  = "posts_tas"
EMBED_MODEL = "text-embedding-3-small"
TOP_K       = 5   # número de fragmentos relevantes a recuperar


class RagInput(BaseModel):
    query: str = Field(description="Tema o pregunta a buscar en los posts de Tech And Solve")


class PostsTASRagTool(BaseTool):
    name: str = "Buscar posts de Tech And Solve"
    description: str = (
        "Busca en la base de conocimiento de posts aprobados de Tech And Solve. "
        "Úsala para conocer el estilo, tono, vocabulario, temas cubiertos y "
        "estructura de los posts reales de la empresa antes de escribir, editar "
        "o sugerir contenido nuevo."
    )
    args_schema: Type[BaseModel] = RagInput

    def _run(self, query: str) -> str:
        # Verificar que el índice existe
        if not CHROMA_DIR.exists():
            return (
                "[RAG] La base de conocimiento aún no está indexada. "
                "Ejecuta 'python knowledge/ingest.py' o usa el botón "
                "'🔄 Actualizar base RAG' en la UI para indexar los posts."
            )

        try:
            import chromadb
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
            col    = chroma.get_or_create_collection(COLLECTION)

            if col.count() == 0:
                return "[RAG] La base de conocimiento está vacía. Indexa los posts primero."

            # Generar embedding de la query
            resp = client.embeddings.create(input=query, model=EMBED_MODEL)
            embedding = resp.data[0].embedding

            # Buscar los TOP_K chunks más similares
            resultados = col.query(
                query_embeddings=[embedding],
                n_results=min(TOP_K, col.count())
            )

            documentos  = resultados.get("documents", [[]])[0]
            metadatos   = resultados.get("metadatas", [[]])[0]

            if not documentos:
                return "[RAG] No se encontraron posts relevantes para esta consulta."

            salida = [f"[Posts de Tech And Solve — relevantes para: '{query}']\n"]
            for i, (doc, meta) in enumerate(zip(documentos, metadatos), 1):
                fuente = meta.get("fuente", "desconocido")
                salida.append(f"--- Fragmento {i} (fuente: {fuente}) ---")
                salida.append(doc[:1500])  # limitar tamaño por chunk
                salida.append("")

            return "\n".join(salida)

        except Exception as e:
            return f"[RAG] Error al consultar la base de conocimiento: {e}"


# Instancia compartida — importar desde aquí en todos los agentes
rag_tool = PostsTASRagTool()
