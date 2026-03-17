from crewai import Task
from agents import topicSuggesterAgent

topicSuggesterTask = Task(
    description="""
        Genera 5 ideas de posts para LinkedIn orientadas al equipo de marketing
        de Tech And Solve. La audiencia objetivo son directivos y líderes del
        sector asegurador en Latinoamérica.

        Para cada idea propuesta debes:
        1. Nombrar el tema con un título atractivo (no genérico).
        2. Explicar POR QUÉ es relevante hoy (tendencia, regulación, dato de mercado).
        3. Sugerir el ángulo o posición que podría tomar Tech And Solve.
        4. Proponer el tipo de contenido: análisis, caso de uso, opinión, dato sorpresa.

        Criterios para seleccionar los temas:
        - Deben estar en tendencia en el sector asegurador / insurtech en LATAM.
        - Deben permitir que Tech And Solve tome una posición clara.
        - Deben ser relevantes para directivos de TI, Operaciones o Marketing de aseguradoras.
        - Evitar temas muy genéricos ("la importancia de la digitalización").
        - Preferir temas con tensión, novedad o datos que sorprendan.
    """,
    agent=topicSuggesterAgent.topicSuggester,
    expected_output="""
        Lista de 5 ideas de posts con el siguiente formato para cada una:

        IDEA #N — [Título atractivo del tema]
        📌 Por qué ahora: [razón de relevancia actual, máx. 2 líneas]
        🎯 Ángulo Tech And Solve: [posición o perspectiva sugerida]
        📝 Tipo de contenido: [análisis / caso de uso / opinión / dato sorpresa]
        ---
    """
)
