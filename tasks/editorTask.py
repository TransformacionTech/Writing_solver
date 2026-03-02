from crewai import Task
from agents import editorAgent

# Define tarea de edición de copy
# Recibe {post} y opcionalmente {feedback} como inputs del kickoff
editCopyTask = Task(
    description = """
        Editar y mejorar el siguiente post de LinkedIn para hacerlo
        más persuasivo, claro y atractivo para aseguradoras.

        Post a mejorar:
        {post}

        {feedback}

        Asegura que sea claro, persuasivo y adecuado para LinkedIn.
        """,
    agent = editorAgent.editor,
    expected_output = """Mejora de 3000 caracteres máximo,
        palabras con enfoque en claridad, persuasión y
        adecuación a LinkedIn, solo entrega el Post mejorado, no agregues nada mas.
        """
)