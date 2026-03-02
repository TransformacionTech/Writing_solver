from crewai import Agent
from customLlm import llm

#Define agente editor del copy
editor = Agent(
    role = "Editor de Copy",
    goal = """
        Analiza, evalua y crea un copy atractivo y persuasivo para la audiencia de LinkedIn, 
        mejorando el texto generado por el escritor para maximizar su impacto comercial.
        """,
    backstory = "Experto en marketing y redacción con años de experiencia",
    verbose = True,
    llm = llm.model #Especificar el llm para este agente
)