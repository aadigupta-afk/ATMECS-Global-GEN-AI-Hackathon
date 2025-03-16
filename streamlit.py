import streamlit as st
from llama_parse import LlamaParse
import google.generativeai as genai
from llama_index.core.schema import TextNode, ImageDocument
from typing import List
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.gemini import Gemini
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize parser and models
parser = LlamaParse(verbose=True)
ollama_mm_llm = GeminiMultiModal(model_name='models/gemini-1.5-flash')
llm = Gemini(model_name='models/gemini-1.5-flash')

Settings.llm = llm
Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

def get_text_nodes(json_list: List[dict]):
    text_nodes = []
    for idx, page in enumerate(json_list):
        text_node = TextNode(text=page["text"], metadata={"page": page["page"]})
        text_nodes.append(text_node)
    return text_nodes

def get_image_text_nodes(json_objs: List[dict]):
    image_dicts = parser.get_images(json_objs, download_path="llama2_images")
    img_text_nodes = []
    for image_dict in image_dicts:
        image_doc = ImageDocument(image_path=image_dict["path"])
        response = ollama_mm_llm.complete(
            prompt="Describe the images as alt text",
            image_documents=[image_doc],
        )
        text_node = TextNode(text=str(response), metadata={"path": image_dict["path"]})
        img_text_nodes.append(text_node)
    return img_text_nodes

def multi_query_retrieval(documents: list, query_engine):
    responses = []
    for ques in documents:
        answer = query_engine.query(ques)
        responses.append(answer)
    return responses

def process_file(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        json_objs = parser.get_json_result(tmp_file_path)
        
        if not json_objs:
            raise ValueError("No content could be extracted from the file.")

        json_list = json_objs[0].get("pages", [])
        
        if not json_list:
            raise ValueError("No pages found in the extracted content.")

        text_nodes = get_text_nodes(json_list)
        image_text_nodes = get_image_text_nodes(json_objs)

        index = VectorStoreIndex(text_nodes + image_text_nodes)
        query_engine = index.as_query_engine()

        os.unlink(tmp_file_path)
        return query_engine
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def main():
    st.title("RAG Application")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "xlsx", "csv"])

    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            query_engine = process_file(uploaded_file)
        
        if query_engine:
            st.success("File processed successfully!")

            user_question = st.text_input("Enter your question:")

            if user_question:
                template = """You are an AI language model assistant. Your task is to generate five 
                different versions of the given user question to retrieve relevant documents from a vector 
                database. By generating multiple perspectives on the user question, your goal is to help
                the user overcome some of the limitations of the distance-based similarity search. 
                Provide these alternative questions separated by newlines. Also you only have to provide the 
                5 questions and no other details in the response. Original question: {question}"""
                prompt_perspectives = ChatPromptTemplate.from_template(template)

                generate_queries = (
                    prompt_perspectives 
                    | ChatOllama(model="llama3.1", temperature=0) 
                    | StrOutputParser() 
                    | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
                )

                retrieval_chain = generate_queries | (lambda x: multi_query_retrieval(x, query_engine))

                rag_template = """Answer the following question based on this context:

                {context}

                Question: {question}
                """

                rag_prompt = ChatPromptTemplate.from_template(rag_template)

                final_rag_chain = (
                    {"context": retrieval_chain, 
                     "question": itemgetter("question")} 
                    | rag_prompt
                    | ChatOllama(model="llama3.1", temperature=0)
                    | StrOutputParser()
                )

                with st.spinner("Generating answer..."):
                    try:
                        answer = final_rag_chain.invoke({"question": user_question})
                        st.write("Answer:", answer)
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
        else:
            st.error("Failed to process the file. Please try again with a different file.")

if __name__ == "__main__":
    main()