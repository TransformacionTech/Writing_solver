# ============================================
# OPENAI WEB SEARCH TOOL — Herramienta de búsqueda
# web usando la Responses API de OpenAI
# ============================================
import os
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)


class WebSearchInput(BaseModel):
    query: str = Field(description="Consulta de búsqueda en internet")


class OpenAIWebSearchTool(BaseTool):
    name: str = "Búsqueda web con OpenAI"
    description: str = (
        "Busca información actualizada en internet usando la búsqueda web nativa de OpenAI. "
        "Úsala para obtener datos recientes, estadísticas, reportes y noticias del sector "
        "asegurador que no estén en tu conocimiento interno."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            response = client.responses.create(
                model="gpt-4o-mini",
                tools=[{"type": "web_search"}],
                input=query,
            )

            # response.output_text es la forma directa según la doc oficial
            result = response.output_text
            if result:
                return result

            return "[Búsqueda web] No se encontraron resultados para la consulta."

        except Exception as e:
            return f"[Búsqueda web] Error al realizar la búsqueda: {e}"


# Instancia compartida
openai_web_search = OpenAIWebSearchTool()
