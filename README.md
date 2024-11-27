"""
# Legacy code
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT}
                                        )

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

    # The LCEL implementation exposes the internals of what's happening around retrieving, 
    formatting documents, and passing them through a prompt to the LLM, but it is more verbose.
    combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)
    chain = create_retrieval_chain(retriever, combine_docs_chain)

"""
