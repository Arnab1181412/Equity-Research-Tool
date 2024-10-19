from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
import os
from dotenv import load_dotenv

encoder = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
llm = ChatGroq(model="llama3-70b-8192", temperature=0.6)
load_dotenv()

ZILLIZ_CLOUD_URI = os.environ["ZILLIZ_CLOUD_URI"]
ZILLIZ_CLOUD_API_KEY = os.environ["ZILLIZ_CLOUD_API_KEY"]


class EquityResearchTool:

    def __init__(self):
        PROMPT_TEMPLATE = """Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
        Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags and avoid using first person pronouns.
        If you don't know the answer, just say that you don't know, don't try to make up an answer or just say you cannot find it if it is really not present there.
        In case answer is not found always generate this text "I could not find the answer in the context provided." and nothing else apart from this at any cost.
        <context>
        {summaries}
        </context>

        <question>
        {question}
        </question>

        The response should be specific and use statistics or numbers when possible.

        Assistant:
        """
        self.prompt = PromptTemplate(template=PROMPT_TEMPLATE,
                                     input_variables=["summaries", "question"])
        index_params = {
            "index_type": 'IVF_FLAT',
            "params": {
                "nlist": 128
            },
            "metric_type": "COSINE"
        }
        self.vector_store = Milvus(embedding_function=encoder,
                                   collection_name="finance_articles",
                                   auto_id=True,
                                   index_params=index_params,
                                   connection_args={
                                       "uri": ZILLIZ_CLOUD_URI,
                                       "token": ZILLIZ_CLOUD_API_KEY,
                                       "secure": True
                                   })

    def add_documents(self, documents):
        self.vector_store.add_documents(documents=documents)

    def answer(self, query, source_urls):
        # for context is provided filter document retrieval by urls provided
        if len(source_urls) != 0:
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                retriever=self.vector_store.as_retriever(
                    search_kwargs={
                        'k': 2,
                        'expr': f"source in {source_urls}"
                    }),
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt})
        # no urls provided
        else:
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                retriever=self.vector_store.as_retriever(
                    search_kwargs={'k': 2}),
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt})

        result = chain(query)
        answer = result['answer']
        if answer != "I could not find the answer in the context provided.":
            if result['sources'] != "":
                return {"answer": answer, "source": result['sources']}
            source = result['source_documents'][0].metadata['source']

            return {"answer": answer, "source": source}
        # retrying a global search
        else:
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                retriever=self.vector_store.as_retriever(
                    search_kwargs={'k': 2}),
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt})

            result = chain(query)
            answer = result['answer']
            # still no relavant answers found
            if answer == "I could not find the answer in the context provided.":
                return {"answer": answer, "source": ""}

            if result['sources'] != "":
                return {"answer": answer, "source": result['sources']}
            source = result['source_documents'][0].metadata['source']

            return {"answer": answer, "source": source}

    def delete_document(self, urls):
        self.vector_store.delete(expr=f"source in {urls}")
