from crewai import Task
from agents import researcherAgent

#Definir tareas
researchTask = Task(
    description = 'Investigar sobre {topic} y encontrar 10 puntos claves',
    agent = researcherAgent.researcher,
    expected_output = 'Lista de 10 puntos relevantes sobre {topic}'
)
