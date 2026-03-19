# ============================================
# DETECTOR DE INTENCIÓN — Writing Solver
# ============================================
# Reemplaza las expresiones regulares por listas de frases
# en texto plano. Para agregar un nuevo trigger basta
# con añadir una línea a la lista correspondiente.
# No se requiere conocimiento de regex.
# ============================================

from typing import Optional

# ─────────────────────────────────────────────────────────────────
# CREAR POST
# Frases que activan el pipeline completo de generación de post.
# El texto que sigue al prefijo se extrae como "tema".
#
# Para agregar una nueva forma de pedir un post, añade el
# prefijo en minúsculas a esta lista.
# ─────────────────────────────────────────────────────────────────
PREFIJOS_CREAR_POST: list[str] = [
    "crea un post sobre",
    "crea un post de",
    "crea un post acerca de",
    "crea un post",
    "crea post sobre",
    "crea post de",
    "genera un post sobre",
    "genera un post de",
    "genera un post acerca de",
    "genera un post",
    "escribe un post sobre",
    "escribe un post de",
    "escribe un post acerca de",
    "escribe un post",
    "hazme un post sobre",
    "hazme un post de",
    "hazme un post",
    "haz un post sobre",
    "haz un post de",
    "haz un post",
    "redacta un post sobre",
    "redacta un post de",
    "redacta un post",
]

# ─────────────────────────────────────────────────────────────────
# SUGERIR TEMAS
# Frases que activan el agente TopicSuggester.
#
# Para agregar una nueva forma de pedir sugerencias, añade
# la frase en minúsculas a esta lista.
# ─────────────────────────────────────────────────────────────────
FRASES_SUGERIR_TEMAS: list[str] = [
    "sugiere temas",
    "sugiere tema",
    "sugerencia de temas",
    "sugerencias de temas",
    "dame ideas",
    "dame temas",
    "dame un tema",
    "qué temas",
    "que temas",
    "ideas de temas",
    "ideas para posts",
    "necesito ideas",
    "propón temas",
    "propone temas",
    "sobre qué puedo escribir",
]


def detectar_tema(mensaje: str) -> Optional[str]:
    """
    Detecta si el usuario quiere crear un post y extrae el tema.

    Compara el mensaje (en minúsculas) contra PREFIJOS_CREAR_POST
    en orden de longitud descendente (el prefijo más largo tiene
    prioridad para evitar coincidencias parciales).

    Returns:
        El tema como string si se detectó un prefijo, None en caso contrario.
    """
    texto = mensaje.strip().lower()

    # Ordenar de más largo a más corto para que prefijos específicos
    # tengan prioridad sobre prefijos genéricos
    for prefijo in sorted(PREFIJOS_CREAR_POST, key=len, reverse=True):
        if texto.startswith(prefijo):
            tema = mensaje.strip()[len(prefijo):].strip()
            return tema if tema else None

    return None


def detectar_sugerencia_temas(mensaje: str) -> bool:
    """
    Detecta si el usuario pide sugerencias de temas para posts.

    Returns:
        True si alguna frase de FRASES_SUGERIR_TEMAS aparece en el mensaje.
    """
    texto = mensaje.strip().lower()
    return any(frase in texto for frase in FRASES_SUGERIR_TEMAS)
