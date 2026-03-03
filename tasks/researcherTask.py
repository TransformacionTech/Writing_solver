from crewai import Task
from agents import researcherAgent

researchTask = Task(
    description="""
        Identifica la temática detrás de la siguiente solicitud del usuario
        y realiza una búsqueda en internet sobre esa temática.

        Solicitud: {topic}

        Ejemplo ilustrativo (solo para entender el formato esperado, no procesar):
        Input: "crea un post con la temática 'factores clave para el éxito
        de un seguro embebido'"
        Temática extraída: "factores clave para el éxito de un seguro embebido"

        Pasos:
        1. Extrae la temática concreta de la solicitud.
        2. Busca información actualizada y relevante en internet.
        3. Entrega los hallazgos con referencias a las fuentes.
    """,
    agent=researcherAgent.researcher,
    expected_output=(
        "La información más valiosa encontrada sobre la temática, "
        "organizada en puntos clave con referencias a las fuentes consultadas."
    ),
)
