from dotenv import load_dotenv
from crewai import LLM
import os

load_dotenv(override=True)  # Cargar variables de entorno desde .env

# ============================================
# CONFIGURACIÓN DEL LLM
# ============================================

model = LLM(
    model='claude-haiku-4-5-20251001',
    temperature=0.7,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

#CONFIGURACIÓN DE OTRO LLM
# anthropic/claude-sonnet-4-6
# claude-haiku-4-5-20251001