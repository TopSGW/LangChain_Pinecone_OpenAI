import os

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENV = os.environ['PINECONE_ENV']

print(">>>>>>>>>>>>> <<<<<<<<<<<<<<")
print(OPENAI_API_KEY)
print(PINECONE_API_KEY)
print(PINECONE_ENV)

loader = TextLoader('../model.txt', encoding='utf8')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

import pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_ENV)

index_name = "xavi-example"

docsearch = Pinecone.from_documents(docs, embedding=embeddings, index_name=index_name)

retriever = docsearch.as_retriever()

qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY), chain_type="stuff", retriever=retriever)

query = "What did the president say about Ketanji Brown Jackson"

print(qa.run(query))

print("-------------doc search-----------------")

docs = docsearch.similarity_search(query)

print(docs[0].page_content)