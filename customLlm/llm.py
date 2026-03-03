from dotenv import load_dotenv
from crewai import LLM
import os

load_dotenv(override=True)

# ============================================
# MODELOS DISPONIBLES
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


# Modelo por defecto
model = create_llm(MODELOS_DISPONIBLES[0])