from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

LLM_MODEL = "microsoft/phi-2"
EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"

QUERY = "Who is Hsun Yu Lee?"
CV_FILE = "CV.pdf"
DB_PATH = "chroma_db"

# ========== Step 1: build LLM ==========
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)

pipe = pipeline("text-generation", model=LLM_MODEL, tokenizer=tokenizer, device=0, max_length=768)     
llm = HuggingFacePipeline(
    pipeline=pipe,
    model_kwargs={
        "temperature": 0.3,
        "max_new_tokens": 256,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "do_sample": True,
    },
)

def wo_RAG():
    print("\n🧪 [只用 LLM 回答]：")
    only_llm_response = llm(QUERY)
    print(only_llm_response)
    print("--------------------------")
def w_RAG():
    # ========== Step 2: build knowledge ==========
    loader = PyPDFLoader(CV_FILE)
    pages = loader.load_and_split()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    embedding = HuggingFaceEmbeddings(model_name=EMBEDDINGS)
    vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=DB_PATH, collection_name='langchain')
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    # if len(vectordb)!= 14, need to remove the folder /chroma_db and run again
    print(f"Total documents in vectordb (need to be 14): {len(vectordb)}")

    # ========== Step 3: build RAG chain ==========
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=qa_chain
    )
    result = chain.invoke({"input": QUERY})

    print("\n🧠 [使用 RAG 回答]：")
    print(result['answer'])


if __name__ == '__main__':
    # without RAG
    wo_RAG()
    # with RAG
    w_RAG()