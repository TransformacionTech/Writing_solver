from crewai import Task
from agents import writerAgent

writerTask = Task(
    description = f"""
        Escribe un post para LinkedIn adoptando el tono de la empresa {writerAgent.tono} 
        que genere conversación e interacción con aseguradoras como 
        clientes potenciales.

        - Los primeros 210 caracteres deben ser un hook fuerte, no cliché
        - Nombra a Tech and Solve naturalmente en el texto
        - porque importa para las aseguradoras (riesgo/impacto en operación 
            y experiencia del usuario/cliente)
        - 2-4 emojis estratégicos para resaltar, no decorar
        - Sin párrafos densos
        - CTA que invite a comentar o agendar conversación con una pregunta
            final potente orientada a riesgo/impacto (no CTA de venta).

        Reglas:
        - No menciones clientes ni personas.
        - Evita tecnicismos; si usas un término, explícalo en una frase.
        - Frases cortas, buen espaciado y lectura fácil.
        - No suene a publicidad.
        """,

    agent = writerAgent.writer,
    expected_output = 'post entre 150-200 palabras'
)