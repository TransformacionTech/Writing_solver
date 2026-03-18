from crewai import Task
from agents import writerAgent

writerTask = Task(
    description="""
Estructura del post que debes escribir:

Gancho (1-2 líneas):
- Debe detener el scroll.
- Puede ser una afirmación provocadora, una estadística inesperada, una pregunta incómoda, o una tensión del sector.
- Evita aperturas genéricas como "En el mundo de los seguros..." o "La transformación digital es..."

Desarrollo (3-5 párrafos cortos):
- Responde literalmente lo que se pidió: si el usuario preguntó "cuáles son", enuméralos con descripción breve de cada uno; si preguntó "qué es", explícalo con ejemplos concretos del sector.
- Usa los datos de la investigación para enriquecer la respuesta, NO para cambiar el tema hacia análisis estratégico o recomendaciones operativas.
- Conecta el insight con las implicaciones prácticas para una aseguradora.
- Cuando uses datos o cifras, cítalos de forma natural e integrada al texto. Ejemplo: "Según McKinsey (2024),..." o "Un estudio de Swiss Re indica que...". No uses notas al pie ni corchetes numéricos: LinkedIn no los renderiza.
- Si la fuente es una organización reconocida por el sector (Swiss Re, Munich Re, MAPFRE, McKinsey, Deloitte, IAIS, FASECOLDA, etc.), inclúyela siempre: aumenta credibilidad ante la audiencia aseguradora.
- No acumules más de 2 citas en el mismo post: afecta la fluidez de lectura en LinkedIn.

Cierre (1-2 líneas):
- Puede ser una reflexión, una pregunta abierta para generar comentarios, o una invitación a pensar diferente.
- Evita cierres tipo "¿Quieres saber más? Contáctanos."

Hashtags:
- Entre 3 y 5, relevantes para el sector y el tema.
- Ejemplos de referencia: #Seguros #TransformaciónDigital #Insurtech #ExperienciaDelAsegurado #Siniestros

---

Aquí están tus inputs:

<solicitud_original>
{topic}
</solicitud_original>

<investigacion>
[La investigación elaborada por el agente investigador está disponible en el contexto de esta tarea]
</investigacion>
    """,
    agent=writerAgent.writer,
    expected_output="""
Responde SIEMPRE con esta estructura exacta:

## POST
[texto del post listo para edición]

## DECISIÓN EDITORIAL
[2-3 líneas explicando si seguiste la solicitud original o tomaste un ángulo distinto, y por qué]

## ALTERNATIVAS DESCARTADAS
[1-2 ángulos que consideraste pero no usaste, para explorar en iteraciones futuras]

## FUENTES UTILIZADAS
[Lista las fuentes citadas en el post con el formato:
Organización/Autor (año si está disponible) — claim específico que respaldaron]

## ALERTAS
[Datos relevantes de la investigación que no pudiste usar por falta de fuente, o problemas estructurales que el editor deba revisar. Si no hay alertas, escribe: "Sin alertas."]
    """,
)