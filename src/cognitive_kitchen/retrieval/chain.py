from typing import List
import functools
import torch
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from cognitive_kitchen.retrieval.vector_store import load_faiss_index

CHEF_PROMPT_TEMPLATE = """You are Chef Pierre, an expert culinary assistant.
Answer the user's recipe question strictly based on the context provided below.
Provide step-by-step instructions. Do not invent instructions outside the context.

Context:
{context}

Question:
{question}

Chef Pierre's Instructions:"""


def format_docs(docs: List[Document]) -> str:
    formatted_chunks = []
    for doc in docs:
        cid = doc.metadata.get("chunk_id", "unknown_chunk")
        formatted_chunks.append(f"[Chunk ID: {cid}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted_chunks)


# In-memory singleton caches to guarantee models load only once in the process
@functools.lru_cache(maxsize=1)
def get_hf_llm(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        device_map="cpu",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=180,
        do_sample=False,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=pipe)


@functools.lru_cache(maxsize=1)
def get_openai_llm(model_name: str = "gpt-4o-mini") -> ChatOpenAI:
    return ChatOpenAI(model=model_name, temperature=0.1, streaming=True)
@functools.lru_cache(maxsize=4)
def get_vector_store(index_name: str):
    return load_faiss_index(index_name=index_name)


def build_chef_chain_with_retriever(
    backend: str = "huggingface",
    index_name: str = "test_naive_experiment",
    k: int = 2,
):
    vector_store = get_vector_store(index_name)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_template(CHEF_PROMPT_TEMPLATE)
    llm = get_openai_llm() if backend == "openai" else get_hf_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever