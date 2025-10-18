#Importing necessary Libraries

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,END
from typing import TypedDict
from dotenv import load_dotenv

#Loding API & create .env file containing the API Key

load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"]=api_key

#Setting up LLM

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3)

#Defining State

class StateAgent(TypedDict):
    legal_clause:str
    clause_explained:str

#Defining the nodes

def get_legal_clause(state:StateAgent)->dict:
    legal_clause=input("Enter the legal clause that you want a simplified explanation for:\n")
    return {"legal_clause":legal_clause}

def explain_legal_clause(state:StateAgent)->dict:
    legal_clause=state["legal_clause"]
    prompt=f"Explain the legal clause in simple English:\n{legal_clause}\n"
    clause_output=llm.invoke(prompt).content
    print("Explanation:\n")
    print(clause_output)
    return {"legal_clause":legal_clause,"clause_explained":clause_output}

#Building Graph

graph=StateGraph(StateAgent)
graph.add_node("getClause",get_legal_clause)
graph.add_node("explainClause",explain_legal_clause)
graph.set_entry_point("getClause")
graph.add_edge("getClause","explainClause")
graph.add_edge("explainClause",END)

app=graph.compile()

#Running the LangGraph Agent

if __name__ == "__main__":
    print("Legal Clause Explainer Agent")

    while True:
        app.invoke({})
        another_clause=input("Do you want to add another clause for explanation? (y/n): ").strip().lower()
        if another_clause!="y":
            print("Visit Again, Bye!")
            break

