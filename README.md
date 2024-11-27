# Motivation

The motivation for this the Edu Tech project is driven by the need to streamline how students access information about the educational tehcnology company, ‘Codebasics’. The project aims to eliminate the need for students to manually search through community resources and the website to find answers. 

# Contribution

Due to frequent updates to the Langchain framework, the system requires ongoing adjustments. A notable change is the rewrite of the RetrievalQA class into a chain using the LangChain Expression Language (LCEL), a more declarative method that simplifies the composition of chains.  This adaptation, along with numerous other minor modifications, positions us well to accommodate future changes. 

In response to the deprecation of Google Palm, I transitioned to OpenAI’s LLM, as Google’s new model also costs money.

The system was last updated on November 26, 2024.

# Architecture

Embedding a list of frequently asked questions using Hugging Face’s Instructor embeddings into a FAISS vector database. 

By integrating the Langchain with an open ai API, the architecture leverages cosine similarity calculations to match student queries with the most relevant questions. 

This ensures that students receive the most relevant answers quickly, enhancing their learning experience and satisfaction with the ‘Codebasics’ educational platform.

For more detail on the architecture, refer to https://www.youtube.com/watch?v=AjQPRomyd-k.


Examples of the modified codes.
"""
    #Legacy code (before)
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT}
                                        )

    #LCEL without the internals (after)
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

    # The LCEL implementation exposes the internals of what's happening around retrieving (after) 
    formatting documents, and passing them through a prompt to the LLM, but it is more verbose.
    combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)
    chain = create_retrieval_chain(retriever, combine_docs_chain)

"""
