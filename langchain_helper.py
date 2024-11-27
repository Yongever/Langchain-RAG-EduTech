from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAI

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"


def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt", encoding="ISO-8859-1")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

    # # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain(llm):
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization="True")

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)


    prompt_template = """Given the following context and a input, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    INPUT: {input}
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # # Legacy code
    # chain = RetrievalQA.from_chain_type(llm=llm,
    #                                     chain_type="stuff",
    #                                     retriever=retriever,
    #                                     input_key="query",
    #                                     return_source_documents=True,
    #                                     chain_type_kwargs={"prompt": PROMPT}
    #                                     )

    #LCEL without the internals
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    #The LCEL implementation exposes the internals of what's happening around retrieving, 
    # formatting documents, and passing them through a prompt to the LLM, but it is more verbose.
    # combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)
    # chain = create_retrieval_chain(retriever, combine_docs_chain)

    return chain

# if __name__ == "__main__":
    # create_vector_db()
    # chain = get_qa_chain(llm)        
    # print(chain)

    #LCEL without the internals
    # print(chain.invoke("Do you have javascript course?"))
    # print(chain.invoke("Do you provide internships?"))
    # print(chain.invoke("what kind of internship do you provide?"))


    #LCEL with the internals
    # print(chain.invoke({"input": "Do you have javascript course?"}))
    # print(chain.invoke({"input": "Do you provide internships?"}))








# test
# from InstructorEmbedding import INSTRUCTOR
# instructor_embeddings = INSTRUCTOR('hkunlp/instructor-large')

# sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
# instruction = "Represent the Science title:"
# embeddings = instructor_embeddings.encode([[instruction,sentence]])
# print(embeddings)