from dotenv import load_dotenv
from crewai import LLM
import os

load_dotenv(override=True)

# ============================================
# MODELOS DISPONIBLES
# ============================================
MODELOS_DISPONIBLES = [
    "claude-haiku-4-5-20251001",
    "anthropic/claude-sonnet-4-6"
]


def create_llm(model_name: str) -> LLM:
    """Crea una instancia del LLM con el modelo especificado."""
    return LLM(
        model=model_name,
        temperature=0.7,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )


# Modelo por defecto
model = create_llm(MODELOS_DISPONIBLES[0])