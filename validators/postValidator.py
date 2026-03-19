import re
from pydantic import BaseModel, Field, model_validator


class ReaderOutput(BaseModel):
    """
    Modelo Pydantic que valida el output del agente Reader.
    Extrae la calificación numérica del texto y determina si el post fue aprobado.

    El Reader puede expresar la calificación de varias formas:
      - "Calificación: 8/10"
      - "8/10"
      - "Puntaje: 7"
      - "Nota: 9"
      - "Score: 8"
    Todos esos formatos son capturados por los patrones definidos abajo.
    """
    texto_completo: str = Field(..., description="Texto completo de la evaluación del Reader")
    calificacion: int = Field(default=0, ge=1, le=10, description="Calificación numérica del 1 al 10")
    aprobado: bool = Field(default=False, description="True si la calificación es >= 8")

    @model_validator(mode='after')
    def extraer_y_validar_calificacion(self) -> 'ReaderOutput':
        """Extrae la calificación del texto y determina si el post está aprobado."""

        # ── Patrones de extracción ────────────────────────────────────────────
        # Cada patrón tiene un nombre descriptivo para facilitar el mantenimiento.
        # Para agregar un nuevo formato de calificación, añade una tupla (nombre, patrón).
        # El primer grupo de captura () siempre debe ser el número.
        patrones: list[tuple[str, str]] = [
            # Ej: "Calificación: 8/10" o "calificacion: 9 / 10"
            ("calificacion_sobre_10",  r'calificaci[oó]n[:\s]+(\d+)\s*/\s*10'),

            # Ej: "8/10" o "9 / 10" en cualquier parte del texto
            ("fraccion_sobre_10",      r'(\d+)\s*/\s*10'),

            # Ej: "Puntaje: 7"
            ("puntaje",                r'puntaje[:\s]+(\d+)'),

            # Ej: "Nota: 9"
            ("nota",                   r'nota[:\s]+(\d+)'),

            # Ej: "Score: 8"
            ("score",                  r'score[:\s]+(\d+)'),

            # Número 8, 9 o 10 aislado (sin letras alrededor)
            # — fallback de último recurso para textos poco estructurados
            ("numero_aislado_alto",    r'\b([89]|10)\b(?!\s*[a-zA-Z])'),
        ]

        texto_lower = self.texto_completo.lower()
        for _nombre, patron in patrones:
            match = re.search(patron, texto_lower, re.IGNORECASE)
            if match:
                try:
                    calificacion = int(match.group(1))
                    if 1 <= calificacion <= 10:
                        self.calificacion = calificacion
                        break
                except (ValueError, IndexError):
                    continue

        self.aprobado = self.calificacion >= 8
        return self


def parsear_output_reader(texto: str) -> ReaderOutput:
    """
    Función de utilidad para parsear el output del Reader.

    Args:
        texto: Texto completo devuelto por el agente Reader.

    Returns:
        ReaderOutput con la calificación extraída y el flag de aprobación.
    """
    return ReaderOutput(texto_completo=texto)
