from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from utils import get_vector_store
from dotenv import load_dotenv

load_dotenv()

def format_docs(docs):
    """Retrieved documents ko ek string mein combine karta hai"""
    return "\n\n".join(doc.page_content for doc in docs)

def chat_with_agent():
    print("\n=== RAG Agent Starting ===")
    vector_store = get_vector_store()

    # Retriever - top 5 relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Groq LLM - FREE aur bahut fast!
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    # Prompt Template
    prompt = ChatPromptTemplate.from_template("""
Aap ek helpful AI assistant hain. Neeche diye gaye context ke basis par question ka jawab dijiye.
Agar jawab context mein nahi hai toh boliye: "Mujhe is topic par documents mein koi information nahi mili."
Jawab concise aur clear hona chahiye. Question jis language mein ho, usi mein jawab dein.

Context:
{context}

Question: {question}

Jawab:""")

    # LCEL Pipeline
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\nAgent ready hai! Apna sawaal poochiye. (Band karne ke liye 'exit' type karein)\n")
    print("-" * 50)

    while True:
        question = input("\nAapka Sawaal: ").strip()
        if not question:
            continue
        if question.lower() in ['exit', 'quit', 'band karo']:
            print("Agent band ho raha hai. Alvida!")
            break

        print("\nSoch raha hoon...")

        docs = retriever.invoke(question)
        answer = rag_chain.invoke(question)

        print("\nAgent ka Jawab:", answer)

        sources = set([doc.metadata.get("source", "Unknown") for doc in docs])
        print(f"\n[📁 Source: {', '.join(sources)}]")
        print("-" * 50)

if __name__ == "__main__":
    chat_with_agent()
