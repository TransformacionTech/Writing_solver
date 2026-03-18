from dotenv import load_dotenv
from crewai import LLM
import os

load_dotenv(override=True)

# ============================================
# MODELOS DISPONIBLES (selector UI)
# ============================================
MODELOS_DISPONIBLES = [
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.2"
]


def create_llm(model_name: str) -> LLM:
    """Crea una instancia del LLM con el modelo especificado."""
    return LLM(
        model=model_name,
        api_key=os.getenv("OPENAI_API_KEY")
    )


# ============================================
# LLMs POR ROL — cada agente usa el modelo
# óptimo para su función
# ============================================

# Rápido y suficiente: procesa resultados de búsqueda web,
# no necesita creatividad
llm_researcher = create_llm("gpt-4o-mini")

# Alta creatividad: escritura de posts con posicionamiento de marca
llm_writer = create_llm("gpt-5.2")

# Creatividad para edición: mejora estructura, tono y persuasión
llm_editor = create_llm("gpt-5.2")

# Algo de creatividad: evalúa con criterio editorial, pero sin generar contenido
llm_reader = create_llm("gpt-4o-mini")

# Chat general y sugeridor de temas
llm_default = create_llm("gpt-4o-mini")

# Alias legacy (para compatibilidad con módulos que importen `llm.model`)
model = llm_default