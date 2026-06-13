from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()  # loads .env file

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
response = llm.invoke("Explain fraud detection in banking in 2 lines.")
print(response.content)