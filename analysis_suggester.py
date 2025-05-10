# analysis_suggester.py
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


@tool
def suggest_analyses(schema: str) -> str:
    """
    Given a schema summary (e.g., "id(INTEGER), headline(STRING), ts(TIMESTAMP)"),
    ask an LLM to propose up to 4 relevant analyses. Each idea on its own line.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior data scientist. "
         "Given the column names and types, list up to four insightful analyses "
         "the user could run next. Return each idea on its own line starting with a dash."),
        ("human", "{schema}")
    ])
    llm = ChatOllama(model="llama3:8b", temperature=0.3)
    return llm.invoke(prompt.format(schema=schema)).content.strip()
