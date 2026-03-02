import re
from pydantic import BaseModel, Field, model_validator

class ReaderOutput(BaseModel):
    """
    Modelo Pydantic para validar el output del agente Reader.
    Extrae la calificación numérica del texto y valida que sea >= 8.
    """
    texto_completo: str = Field(..., description="Texto completo de la evaluación del Reader")
    calificacion: int = Field(default=0, ge=1, le=10, description="Calificación numérica del 1 al 10")
    aprobado: bool = Field(default=False, description="True si la calificación es >= 8")

    @model_validator(mode='after')
    def extraer_y_validar_calificacion(self) -> 'ReaderOutput':
        """Extrae la calificación del texto y determina si el post está aprobado."""
        # Patrones para detectar calificación en el texto
        patrones = [
            r'calificaci[oó]n[:\s]+(\d+)\s*/\s*10',
            r'(\d+)\s*/\s*10',
            r'puntaje[:\s]+(\d+)',
            r'nota[:\s]+(\d+)',
            r'score[:\s]+(\d+)',
            r'\b([89]|10)\b(?!\s*[a-zA-Z])',  # Número 8, 9 o 10 aislado
        ]

        texto_lower = self.texto_completo.lower()
        for patron in patrones:
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
        texto: Texto completo devuelto por el agente Reader
    
    Returns:
        ReaderOutput con la calificación extraída y el flag de aprobación
    """
    return ReaderOutput(texto_completo=texto)
