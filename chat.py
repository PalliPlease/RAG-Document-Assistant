
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM as Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

#VectorDB storage location
PERSIST_DIR = "vectordb"


def getvector(): #returns a chroma vector database instance
    embeddings = OllamaEmbeddings(model="nomic-embed-text") #using ollama to convert text to vectors

    #create a chroma vectordb, which is our retriever's storage
    vectordb = Chroma(
        embedding_function = embeddings,
        persist_directory=PERSIST_DIR
    )
    return vectordb

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_qa_chain():
    vectordb = getvector() #load vectordb
    retriever = vectordb.as_retriever( #create a retriever
        search_type="similarity",
        search_kwargs={"k":2} #edit this to speed up (finds the top 2 most similar docs)
    )

    llm=Ollama(
        model="llama3",
        temperature=0.1 #make it factual and less craetive
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant for answering questions about a college handbook.
        Use ONLY the provided context to answer. If the answer is not in the context,
        say you don't know and suggest where the student might check (e.g., admin office, website).
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer in a clear, concise way, suitable for a student."""
    )

    qa_chain = (
        {
            "context": retriever | format_docs, #take retrievers output and give it to format_docs
            "question": RunnablePassthrough(), #pass user query as is
        }
        | prompt
        | llm
        | StrOutputParser() #converts LLM outputs into string
    )

    # return both the chain (for answers) and retriever (for sources)
    return qa_chain, retriever

def main():
    qa_chain, retriever = build_qa_chain()
    print("Handbook assistant (Ollama) ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            print("Bye!")
            break

        # get answer from the chain
        answer = qa_chain.invoke(question)

        print("\nAssistant:", answer)


if __name__ == "__main__":
    main()