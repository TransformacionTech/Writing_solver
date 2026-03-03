from crewai import Task
from agents import readerAgent

# Define la tarea de lectura y análisis del copy
readerTask = Task(
    description = """
        Evalúa el copy generado usando los criterios editoriales exactos de
        Tech And Solve. Tu evaluación debe ser objetiva, basada en criterios
        verificables, no en impresiones generales y teniendo en cuenta lo que
        pide el usuario.

        CRITERIOS DE EVALUACIÓN (cada uno tiene peso):

        [A] APERTURA — ¿Las primeras 4 líneas generan lectura inmediata?
            ✅ Hook que provoca o tensiona
            ✅ Sin introducir contexto, justificaciones ni explicaciones
            ✅ Una sola idea fuerte que jala hacia el cuerpo
            ❌ Penaliza: comenzar con "En Tech And Solve..." como primera línea,
                preguntas retóricas débiles, frases de autoempaque

        [B] VOZ DE MARCA — ¿Respeta el tono Tech And Solve?
            ✅ Cercano, humano, profesional sin rigidez
            ✅ Inspirador con propósito, no motivacional vacío
            ✅ Coloca al lector en el centro, no a la empresa
            ❌ Penaliza: tono corporativo plano, clichés del sector,
                mencionar lo que "no somos", diminutivos, minimizar competidores

        [C] IMPACTO COMERCIAL — ¿Abre una puerta a conversación?
            ✅ El lector siente que esto le habla a él/ella directamente
            ✅ Demuestra experiencia sin declarar explícitamente el valor
            ✅ Genera tensión, reflexión o deseo de contacto
            ❌ Penaliza: post meramente informativo, ausencia de intención
                comercial, listas sin impacto narrativo

        [D] REGLAS EDITORIALES DURAS — Verificación objetiva:
            ✅ Longitud entre 200 y 250 palabras
            ✅ Emojis estratégicos (no decorativos ni excesivos)
            ✅ Vocabulario variado, sin repetición ni clichés
            ✅ No menciona cargos ni valores explícitos ("somos los mejores",
                precios, métricas que no están en el brief)
            ✅ Si menciona a Tech And Solve, lo hace de forma natural, no forzada
            ❌ Penaliza cada violación de regla dura con -1 punto

        [E] CIERRE — ¿Deja la sensación correcta?
            ✅ Frase memorable, reflexión, invitación suave o CTA sutil
            ✅ Incluso sin CTA explícito: ¿el lector siente que "esto merece conversación"?
            ❌ Penaliza: cierres genéricos tipo "¡Contáctanos!", promesas vacías
                """,
    agent = readerAgent.reader,
    expected_output = """
        Devuelve tu evaluación con el siguiente formato EXACTO:

        Calificación: X/10

        🔍 Criterios evaluados:
        - [A] Apertura: X/2 — [razón breve]
        - [B] Voz de marca: X/2 — [razón breve]
        - [C] Impacto comercial: X/2 — [razón breve]
        - [D] Reglas editoriales: X/2 — [indica si hay violaciones específicas]
        - [E] Cierre: X/2 — [razón breve]

        ✅ Qué funciona: [1 oración concreta]
        ⚠️ Qué no funciona: [1 oración concreta con el problema principal]
        💡 Mejora sugerida: [1 acción concreta y aplicable, no genérica]

        Veredicto: APROBADO / RECHAZADO
        (El copy debe obtener mínimo 8/10 Y no tener ninguna violación de regla
        dura en [D] para ser APROBADO. Un 8/10 con violaciones en [D] = RECHAZADO.)
    """
)
