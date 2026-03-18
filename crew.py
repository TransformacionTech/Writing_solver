# ============================================
# CREW - Sistema Multi-Agente con Gradio
# ============================================
import re
import gradio as gr
from crewai import Crew
from agents import researcherAgent, writerAgent, editorAgent, readerAgent, chatAgent, topicSuggesterAgent
from tasks import researcherTask, writerTask, editorTask, readerTask, topicSuggesterTask
from validators.postValidator import parsear_output_reader
from customLlm.llm import create_llm, MODELOS_DISPONIBLES

# ─────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────
MAX_REINTENTOS = 3
TRIGGER = re.compile(r"crea\s+un\s+post\s+(.+)", re.IGNORECASE)
TRIGGER_TEMAS = re.compile(r"sugiere\s+temas?|dame\s+ideas?|qu[eé]\s+temas?", re.IGNORECASE)

TODOS_LOS_AGENTES = [
    researcherAgent.researcher,
    writerAgent.writer,
    editorAgent.editor,
    readerAgent.reader,
    chatAgent.assistant,
    topicSuggesterAgent.topicSuggester
]


def actualizar_llm(model_name: str) -> None:
    """Reasigna el LLM a todos los agentes con el modelo seleccionado."""
    nuevo_llm = create_llm(model_name)
    for agente in TODOS_LOS_AGENTES:
        agente.llm = nuevo_llm

# ─────────────────────────────────────────────
# Detección de intención
# ─────────────────────────────────────────────

def detectar_sugerencia_temas(mensaje: str) -> bool:
    """Devuelve True si el usuario pide sugerencias de temas."""
    return bool(TRIGGER_TEMAS.search(mensaje.strip()))


def run_topic_pipeline() -> str:
    """Ejecuta el agente sugeridor de temas y devuelve las 5 ideas."""
    crew_temas = Crew(
        agents=[topicSuggesterAgent.topicSuggester],
        tasks=[topicSuggesterTask.topicSuggesterTask],
        verbose=False
    )
    return str(crew_temas.kickoff())


def detectar_tema(mensaje: str) -> str | None:
    """
    Devuelve el tema si el usuario quiere crear un nuevo post,
    o None si es una conversación normal sobre el post existente.
    Ejemplos que activan el pipeline:
      - "Crea un post sobre APIs abiertas"
      - "Crea un post modernización digital"
    """
    match = TRIGGER.search(mensaje.strip())
    return match.group(1).strip() if match else None


# ─────────────────────────────────────────────
# Pipeline de generación del post
# ─────────────────────────────────────────────

def run_pipeline(tema: str) -> tuple[str, list[str]]:
    """
    Flujo completo usando las tareas definidas en tasks/:
      Fase 1: researcherTask → writerTask   (una sola vez)
      Fase 2: editorTask → readerTask       (con re-intentos si calificación < 8)
    """
    log = []
    feedback_anterior = ""

    # ── Fase 1: Investigación + Escritura ────────────────────────────────────
    # writerTask usa context=[researcherTask] para recibir el output del Researcher
    writerTask.writerTask.context = [researcherTask.researchTask]

    crew_fase1 = Crew(
        agents=[researcherAgent.researcher, writerAgent.writer],
        tasks=[researcherTask.researchTask, writerTask.writerTask],
        verbose=True
    )
    resultado_fase1 = crew_fase1.kickoff(inputs={"topic": tema})
    post_actual = str(resultado_fase1)

    log.append(f"🔍 Investigación completada sobre: _{tema}_")
    log.append("✍️ Post inicial generado por el Writer")

    # ── Fase 2: Edición + Evaluación (con re-intentos) ───────────────────────
    # readerTask usa context=[editorTask] para recibir el post del Editor
    readerTask.readerTask.context = [editorTask.editCopyTask]

    # El Crew se construye UNA SOLA VEZ fuera del loop para evitar
    # overhead de instanciación en cada iteración
    crew_fase2 = Crew(
        agents=[editorAgent.editor, readerAgent.reader],
        tasks=[editorTask.editCopyTask, readerTask.readerTask],
        verbose=True
    )

    for intento in range(1, MAX_REINTENTOS + 1):

        # Construir el texto de feedback para inyectar en la tarea del editor
        texto_feedback = ""
        if feedback_anterior:
            texto_feedback = (
                f"Feedback del Reader del intento anterior (incorpóralo en tu mejora):\n"
                f"{feedback_anterior}"
            )

        crew_fase2.kickoff(inputs={
            "topic": tema,
            "post": post_actual,
            "feedback": texto_feedback
        })

        texto_evaluacion = str(readerTask.readerTask.output.raw) if readerTask.readerTask.output else ""
        post_editado = str(editorTask.editCopyTask.output.raw) if editorTask.editCopyTask.output else post_actual

        evaluacion = parsear_output_reader(texto_evaluacion)
        calificacion = evaluacion.calificacion

        log.append(
            f"✏️ Edición #{intento} — "
            f"Reader: {calificacion}/10 "
            f"{'✅ Aprobado' if evaluacion.aprobado else '🔄 Re-intentando...'}"
        )

        if evaluacion.aprobado:
            return post_editado, log

        feedback_anterior = texto_evaluacion
        post_actual = post_editado

    log.append(f"⚠️ Máximo de {MAX_REINTENTOS} intentos. Devolviendo mejor versión.")
    return post_actual, log


# ─────────────────────────────────────────────
# Agente conversacional (chat sobre el post)
# ─────────────────────────────────────────────

def responder_sobre_post(mensaje: str, post_contexto: str, history: list) -> str:
    """
    Usa chatAgent para responder preguntas o peticiones sobre el post ya generado.
    """
    from crewai import Task

    historial_texto = ""
    for msg in history[-8:]:
        rol = "Usuario" if msg["role"] == "user" else "Asistente"
        contenido = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
        historial_texto += f"{rol}: {contenido}\n"

    tarea_chat = Task(
        description=f"""
            Tienes acceso al post de LinkedIn aprobado por el equipo.

            Post actual:
            {post_contexto}

            Historial reciente de la conversación:
            {historial_texto}

            Petición del usuario: {mensaje}

            Instrucciones:
            - Si el usuario pide modificar el post, devuelve el post COMPLETO modificado.
            - Si el usuario hace una pregunta, respóndela brevemente.
            - Mantén siempre el tono de Tech And Solve: {writerAgent.tono}
            - No generes un nuevo post desde cero a menos que el usuario lo pida explícitamente.
        """,
        agent=chatAgent.assistant,
        expected_output="Respuesta directa al usuario. Si hay post modificado, incluirlo completo."
    )

    crew_chat = Crew(
        agents=[chatAgent.assistant],
        tasks=[tarea_chat],
        verbose=False
    )
    return str(crew_chat.kickoff())


# ─────────────────────────────────────────────
# Lógica principal de chat (router)
# ─────────────────────────────────────────────

def chat(mensaje: str, history: list, post_state: str, modelo: str) -> tuple[str, str]:
    """
    Router:
    - "Crea un post [tema]" → ejecuta el pipeline completo
    - Cualquier otro mensaje → chatAgent responde con el contexto del post
    """
    if not mensaje.strip():
        return "Por favor escribe un mensaje.", post_state

    actualizar_llm(modelo)

    # ── Modo: SUGERIR TEMAS ──────────────────────────────────────────────────
    if detectar_sugerencia_temas(mensaje):
        try:
            ideas = run_topic_pipeline()
            respuesta = (
                f"### 💡 Ideas de Posts para Marketing\n\n{ideas}\n\n"
                f"---\n📌 Di **\"Crea un post [tema]\"** para generar cualquiera de estas ideas."
            )
            return respuesta, post_state
        except Exception as e:
            return f"❌ **Error al sugerir temas:**\n\n`{str(e)}`", post_state

    tema = detectar_tema(mensaje)

    # ── Modo: CREAR POST ─────────────────────────────────────────────────────
    if tema:
        try:
            post_final, log = run_pipeline(tema)
            proceso = "\n".join(log)
            respuesta = (
                f"### 📋 Proceso completado\n{proceso}\n\n"
                f"---\n\n"
                f"### 📝 Post Final Aprobado\n\n{post_final}\n\n"
                f"---\n💬 Ahora puedes pedirme ajustes sobre el post, o hacer preguntas sobre él."
            )
            return respuesta, post_final

        except Exception as e:
            return f"❌ **Error en la generación:**\n\n`{str(e)}`", post_state

    # ── Modo: CONVERSACIÓN sobre el post ─────────────────────────────────────
    if not post_state:
        return (
            'Aún no hay un post generado. '
            'Di **"Crea un post [tema]"** para que los agentes lo generen.',
            post_state,
        )

    try:
        respuesta = responder_sobre_post(mensaje, post_state, history)
        nuevo_post_state = respuesta if len(respuesta) > 100 else post_state
        return respuesta, nuevo_post_state

    except Exception as e:
        return f"❌ **Error en el chat:**\n\n`{str(e)}`", post_state


# ─────────────────────────────────────────────
# Interfaz Gradio con gr.Blocks y gr.State
# ─────────────────────────────────────────────

with gr.Blocks(title="✍️ Writing Solver – Tech And Solve") as demo:

    post_state = gr.State("")

    gr.Markdown(
        """
        # ✍️ Writing Solver – Tech And Solve
        Genera y refina posts de LinkedIn con un equipo de agentes de IA.

        **Para crear un post:** escribe `Crea un post [tema]`
        > Ejemplo: *Crea un post sobre APIs abiertas en seguros*

        **Luego puedes pedir ajustes** directamente en el chat, por ejemplo:
        - *"Hazlo más corto"*
        - *"Cambia el hook por algo más directo"*
        - *"¿Cuál es el CTA del post?"*
        """
    )

    with gr.Row():
        modelo_selector = gr.Dropdown(
            choices=MODELOS_DISPONIBLES,
            value=MODELOS_DISPONIBLES[0],
            label="🤖 Modelo",
            scale=2
        )

    chatbot = gr.Chatbot(label="Chat")

    with gr.Row():
        txt = gr.Textbox(
            placeholder='Escribe "Crea un post sobre [tema]" o pide "Sugiere temas"...',
            show_label=False,
            autofocus=True
        )

    with gr.Row():
        btn = gr.Button("Enviar", variant="primary", scale=1)
        btn_temas = gr.Button("💡 Sugerir temas", variant="secondary", scale=1)
        btn_rag = gr.Button("🔄 Actualizar DB", variant="secondary", scale=1)

    gr.Examples(
        examples=[
            "Crea un post sobre modernización de sistemas legacy en aseguradoras",
            "Crea un post sobre DevOps para aseguradoras en LATAM",
            "Crea un post sobre seguros embebidos",
        ],
        inputs=txt,
        label="Ejemplos",
    )

    def responder(mensaje, history, post_state, modelo):
        respuesta, nuevo_state = chat(mensaje, history, post_state, modelo)
        history = history + [
            {"role": "user", "content": mensaje},
            {"role": "assistant", "content": respuesta}
        ]
        return history, nuevo_state, ""

    def sugerir_temas(history, post_state, modelo):
        actualizar_llm(modelo)
        try:
            ideas = run_topic_pipeline()
            respuesta = (
                f"### 💡 Ideas de Posts para Marketing\n\n{ideas}\n\n"
                f"---\n📌 Di **\"Crea un post [tema]\"** para generar cualquiera de estas ideas."
            )
        except Exception as e:
            respuesta = f"❌ **Error al sugerir temas:**\n\n`{str(e)}`"
        history = history + [
            {"role": "user", "content": "💡 Sugerir temas para marketing"},
            {"role": "assistant", "content": respuesta}
        ]
        return history, post_state

    btn.click(
        fn=responder,
        inputs=[txt, chatbot, post_state, modelo_selector],
        outputs=[chatbot, post_state, txt]
    )
    txt.submit(
        fn=responder,
        inputs=[txt, chatbot, post_state, modelo_selector],
        outputs=[chatbot, post_state, txt]
    )
    btn_temas.click(
        fn=sugerir_temas,
        inputs=[chatbot, post_state, modelo_selector],
        outputs=[chatbot, post_state]
    )

    def actualizar_rag(history, post_state):
        """Ejecuta ingest.py para reindexar los posts de post/ en ChromaDB."""
        import subprocess, sys
        try:
            resultado = subprocess.run(
                [sys.executable, "knowledge/ingest.py"],
                capture_output=True, text=True, timeout=120
            )
            salida = resultado.stdout.strip() or resultado.stderr.strip()
            if resultado.returncode == 0:
                respuesta = f"✅ **Base RAG actualizada.**\n\n```\n{salida}\n```\n\nLos agentes ya pueden consultar los posts indexados."
            else:
                respuesta = f"❌ **Error al actualizar la base RAG:**\n\n```\n{salida}\n```"
        except Exception as e:
            respuesta = f"❌ **Error inesperado:**\n\n`{str(e)}`"
        history = history + [
            {"role": "user", "content": "🔄 Actualizar base RAG"},
            {"role": "assistant", "content": respuesta}
        ]
        return history, post_state

    btn_rag.click(
        fn=actualizar_rag,
        inputs=[chatbot, post_state],
        outputs=[chatbot, post_state]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)