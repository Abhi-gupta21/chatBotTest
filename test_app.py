import os
import streamlit as st 
from pydub import AudioSegment
import torch
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
import faiss
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from dotenv import load_dotenv
# load_dotenv()

# setting up the env vars
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"] or os.getenv("OPENAI_API_KEY")
os.environ['LANGCHAIN_API_KEY'] = st.secrets["LANGCHAIN_API_KEY"] or os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = st.secrets["LANGCHAIN_PROJECT"] or os.getenv("LANGCHAIN_PROJECT")
os.environ['OPENAI_API_MODEL'] = 'gpt-4o'

st.title("CCHMC GUIDE BOT")



# my llm model 
llm = ChatOpenAI(model="gpt-4o")
print(llm)
# embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions = 1024)
# database connection
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
print(db)
retriever = db.as_retriever()
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
retrieval_chain = create_retrieval_chain(retriever, document_chain)

def query_gpt4(input):
    answer = retrieval_chain.invoke({"input": input})
    return answer


# user input
text_input = st.text_input("what is your query?")
audio_input = st.audio_input("audio input")
language = st.selectbox("language options", ['english', 'spanish', 'hindi']) or 'english'


if audio_input:
    st.audio(audio_input)
    audio_wav = AudioSegment.from_wav(audio_input)
    audio_wav.export("audio.mp3", format="mp3")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )



    result = pipe("audio.mp3", generate_kwargs={"task": "translate"})
    st.write(result["text"])

    input = result["text"]
    answer = query_gpt4(input)
    st.write(answer['answer'])

elif text_input:
    input = text_input
    answer = query_gpt4(input)
    st.write(answer['answer'])

else:
    st.write("No input provided")

# answer = query_gpt4(input)
# st.write(answer['answer'])


# from googletrans import Translator
# translator = Translator()

# dest = 'en'

# if language: 
#     if language == 'spanish':
#         dest = 'es'
#     elif language == 'hindi':
#         dest = 'hi'

# async def translation(text, dest):
#     return await translator.translate(text, dest=dest).text


# if input:
#     answer = retrieval_chain.invoke({"input": input})
#     translated_answer = translation(answer['answer'], dest)
#     print(answer['answer'])
#     st.write(translated_answer)


