from crewai import Agent
from customLlm import llm

#Define agente investigador
researcher = Agent(
    role = 'Investigador Senior',
    goal = 'Descubir información relevante sobre {topic}',
    backstory = 'Experto en investigación con años de experiencia',
    verbose = False,
    llm = llm.model #Especificar el llm para este agente
)