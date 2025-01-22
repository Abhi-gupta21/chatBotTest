import os
import streamlit as st 
# from dotenv import load_dotenv
# load_dotenv()

# setting up the env vars
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"] or os.getenv("OPENAI_API_KEY")
os.environ['LANGCHAIN_API_KEY'] = st.secrets["LANGCHAIN_API_KEY"] or os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = st.secrets["LANGCHAIN_PROJECT"] or os.getenv("LANGCHAIN_PROJECT")

st.title("CCHMC GUIDE BOT")

# user input
input = st.text_input("what is your query?")

# my llm model 
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
print(llm)



# embeddings
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions = 1024)



# database connection
from langchain_community.vectorstores import FAISS
import faiss

db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

print(db)

retriever = db.as_retriever()

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# prompt
prompt = ChatPromptTemplate.from_template(
    """
Answer the question/question's given by the user based on the provided context:
<context>
{context}
</context>

Question: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)

print(document_chain)

from langchain.chains import create_retrieval_chain


retrieval_chain = create_retrieval_chain(retriever, document_chain)

if input:
    answer = retrieval_chain.invoke({"input": input})
    print(answer)
    st.write(answer)


