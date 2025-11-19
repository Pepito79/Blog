---
title: Create a RAG with LangChain and ChromDB
draft: false
tags:
  - GenAI
  - Python
---
# Introduction

After the fine-tuning arc, I decided to dive into how RAGs work and how to implement them.

**What is a RAG?**

RAG stands for *Retrieval-Augmented Generation*. It allows you to add external context and knowledge to an LLM because an LLM’s knowledge is limited to what was included in its training data. A RAG lets us integrate external data such as PDFs, images, or other types of documents. On top of that, it was the perfect concept for me to get started with LangChain.

# How does a RAG work?

There are 5 main steps we need to understand to build a RAG:

1. **Loading:** load your documents (PDFs, images, CSVs…)
2. **Splitting and chunking:** split these documents into smaller chunks  
3. **Embedding:** embed every chunk we generated  
4. **Storing:** store these embeddings in a vector database  
5. **Retrieving:** use the vector store as a retriever, allowing the LLM to search for relevant information to answer the user’s query.

If you didn’t understand every word, don’t worry — I’ll explain everything.

But before diving deeper, I personally like having a global idea of how a concept works. So here’s a quick overview:

- Let’s say the input documents are PDFs. We load them and split them into smaller units: chunks.
- Then we take each chunk and convert it into a vector of floats — this is called embedding.
- We store these vectors in a vector database (a vector DB is extremely fast).

After these steps, when a user sends a query, the input text is converted into a vector (**an embedding**). Then we use the vector DB to find the closest vectors (closest in meaning). Once we find them, we convert those chunks back into text and send them to the LLM along with the user’s request. This adds context to the LLM, which can now answer with this new knowledge.

# 1) Loading and chunking documents

To start, I’ll only load a PDF to see how everything works. For this, we’ll need the `text_loader` module from the `langchain_community` framework.

I decided to use the PyMuPDF framework because it extracts text with structure (headers, lists, etc.) and can convert it to Markdown — which is LLM-friendly and helps produce better results.


```python
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from typing import List
import torch
import threading
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough , RunnableLambda
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import hashlib
from transformers import AutoTokenizer
```

The thing here is that we want to extract all the pdf documents in a folder , it means that even if the folder contains others directories where there is pdf document we would be able to find them and use them. I wanted to do this to give the user more flexibility and allow him to organize his folder as he wants.

  
The goal here is to extract all the PDF documents inside a folder. This means that even if the folder contains subdirectories with PDF files, we still want to be able to find and use them. I chose this approach to give the user more flexibility and allow them to organize their folders however they want.

To do this, we use the `DirectoryLoader` class, which allows us to specify:

- **Path** — the root directory to search
- **Glob pattern** — to match specific files (e.g., `**/*.pdf`)
- **PdfLoader** — the loader used to read PDF files



```python
separators=["\n\n", "\n", " ", "","\n\n"]
def load_docs(path:str):
 """
 **path**  
 path where all the pdf are stocked  
 **returns** : List[Document] , int
 """
 assert os.listdir(path) , "The path does not contain any pdf"
 loader = DirectoryLoader(
    path= path,
    glob = "**/*.pdf",
    show_progress=True,
    recursive= True,
 )
 docs = loader.load()
 print(f" Successfully loaded {len(docs)} dpcs")
 return docs 
```


```python
def split_document(path: str):
    #We load our documents 
    docs = load_docs(path)
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
    #Here we split our text by tokens and not simply 
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer= tokenizer,
        chunk_size = 512,
        chunk_overlap = int(512/10),
        add_start_index = True,               # We save the index of each chunk 
        strip_whitespace = True ,       
        separators = separators
    )
    chunks = text_splitter.split_documents(docs)  # List[Documents]
    return chunks
```

Okay now we have a list of documents where every document contains two elements:
- Metadata (can be useful)
- Page_content : the content of the chunked part

If you have followed my until here you may understand that we still have 3 steps to finalize our RAG. Let's see the third step
# 2) Step 3 : embbedings

Now that we have our chunks , we want to convert them in vectors of float : these vectors are called **embeddings** . This transformation is done through an **embedding model** that aims to capture the meanning of the text/chunk .

**But why do wee need this ?**

We need this because we want to store the vectors in a Vector Database (VectoreDB) where we will be able to store and retrieve high-dimensional vector data . And what matters is the fact that in this DB vectors with a **close meanning** are located closer together in the vector space and when our RAG app will receives a user input, it will be embedded and used to query the database, returning the most similar documents.

For this I will use the embedding model : `intfloat/multilingual-e5-small`

```python
# This util function will allow us to only load one time the model in our memory to avoid loading it every time 
# I used a lock to avoid race condition do this function
_model_cache = None
_model_lock = threading.Lock()
def get_model():
    global _model_cache
    if _model_cache is None:
        with _model_lock:  # lock for one thread 
            if _model_cache is None:  
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                _model_cache = SentenceTransformer('intfloat/multilingual-e5-small', device=device)
    return _model_cache
def get_embeddings(docs : List[Document]) :
    """ 
    **docs**  
    The list of chunks created by loading the pdfs  
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using cuda " if device == "cuda" else "Using the cpu")
    model = get_model()
    embeddings = model.encode([doc.page_content for doc in docs], convert_to_tensor=True, device=device)
    assert len(embeddings) >0 , "The document is empty: NO EMBEDDING GENERATED"
    print(f"Successfully generated {len(embeddings)} embeddings")
    #Convert embeddings to a list 
    embeddings_list = embeddings.tolist()
    return embeddings_list
```

# 3) Step 4 : Storing embeddings in a VectorDB

For this part, I used the `ChromaDB` framework as the VectorDB. ChromaDB is an open-source AI application database that provides everything we need for retrieval:

- Embedding storage with metadata  
- Vector search  
- Full-text search  
- Document storage  
- Metadata filtering  
- Multi-modal retrieval  

At first, I wrote the code mainly using the ChromaDB library directly. This helped me understand how Chroma works and what its main operations look like.  
However, LangChain also provides a package that integrates Chroma directly: `langchain_chroma`

## **So you’ll find two versions of this step: one using raw Chroma and one using LangChain. Feel free to read whichever you prefer (or both)


# I ) ChromaDB implementation

```python
def store_embeddings(chromaDbPath: str, name_collection: str, chunks : List[Document]):
    """
    **chromaDbPath** 
    The path of the persistent client , if it does not exists it creates a new one  
    **name_collection**  
    Name wanted to the collection that we want to create to stock our vectors  
    **chunks**  
    List of chunks generated by loading and chunking the documents  
    """
    client = chromadb.PersistentClient(path=chromaDbPath)
    assert client is not None , "Problem in getting/creating the client"
    #Create a collection 
    collection = client.get_or_create_collection(
        name= name_collection,
        embedding_function= None #We will use our embeddings
    )
    # Create Ids and get embeddings of the document
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = get_embeddings(chunks)
    metadata = [chunk.metadata for chunk in chunks]
    documents = [chunk.page_content for chunk in chunks]
    try:
        collection.add(
            ids = ids  ,
            embeddings = embeddings,
            metadatas=metadata,
            documents= documents
        )
    except Exception as e:
        print(" Error while adding to ChromaDB : ",e)
    print("Successfully added all the docs to the vector database")
```


```python
def query_vectorDb(query: str,chromaDbPath: str, collection_name :str,n_results: int) -> str:
    assert len(query) >0 , "Query cannot be empty"
    #Load the client and the db
    client = chromadb.PersistentClient(path= chromaDbPath)
    collection = client.get_collection(
        name= collection_name,
        embedding_function=None
    )
    doc_query = [Document(metadata={},page_content=query)]
    model_name = 'intfloat/multilingual-e5-small'   
    embedded_query = get_embeddings(doc_query,model_name)
    result = collection.query(
        query_embeddings=embedded_query,
        n_results= n_results
    )
    return result
```

Up to this point, we only have access to a set of relevant documents that *might* contain the answer to our query.  
The final step is simple: we give these documents (the chunks) as **context** to an LLM, provide the query, and ask it to answer **using only the given context**.

I didn’t implement this final step here because I preferred to be more efficient and more realistic by rewriting the entire pipeline above using LangChain.

# II) LangChain implementation

Before diving into the code, I’ll define a function that generates deterministic IDs for every embedding.  
We do this because if we give our RAG the same document twice, we don’t want it to recompute the embeddings and create new IDs each time.


```python
def compute_id(doc: Document):
    return hashlib.md5(doc.page_content.encode()).hexdigest()
def store_embeddings_with_langchain(collection_name: str , persist_dir:str , chunks : List[Document] ):
    """ 
    **collection_name**  
    Name of the collection that already exists or to create  
    **persist_dir**  
    Directory where collection and db are stocked  
    **chunks**  
    List of chunks (that under the hood are Documents) generated  
    #**returns**  
    Chroma instance
    """
    if torch.cuda.is_available():
        model_kwargs = {"device": "cuda"}
        print("Using Cuda to generate embeddings")
    else:
        model_kwargs=  {"device": "cpu"}
        print("Using CPU to generate embeddings")
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small", model_kwargs=model_kwargs)
    vector_store = Chroma(
        collection_name= collection_name,
        embedding_function= embeddings,
        persist_directory= persist_dir,
    )
    #Let's check if the the collection already contains the vectors
    current_content = vector_store.get() # It's returns a dic with the ids and the embeddings generated
    existing_ids = set(current_content["ids"]) if current_content["ids"] else set()
    docs_to_add = []
    ids_to_add = []
    for doc in chunks:
        doc_id = compute_id(doc)
        #Verify if the id is alread in the db
        if doc_id not in existing_ids:
            docs_to_add.append(doc)
            ids_to_add.append(doc_id)
    if len(docs_to_add) == 0:
        print("Documents already in the DB nothing will be added to the vector store")
        return vector_store
    try:
        vector_store.add_documents(documents=docs_to_add,ids=ids_to_add)
    except Exception as e:
        print(" Error while adding to ChromaDB : ",e)
    print(f"Successfully added {len(docs_to_add)} vectors in the vector database ")
    return vector_store
```

Now the function to query our vector store becomes much simpler because LangChain handles everything for us through the `Chroma` class.  
We will create a pipeline using one of the most interesting features of LangChain: **chains**, and more precisely, we will create an **LCEL**, which stands for *LangChain Expression Language*.

At this point, you may be wondering: *“He keeps talking about chains, but what exactly are these chains linking together?”*  
The answer is: **runnables**.

My next post will be about Runnables in LangChain, so here I’ll just give a quick explanation of how they work and why they are used.

# Runnables

A **Runnable** in LangChain is the basic building block that can be linked to other Runnable objects.  
In fact, `Runnable` is an abstract class in LangChain, and other components (such as `PromptTemplate`, `ChatOpenAI`, etc.) inherit from it.  
It provides the methods that allow Runnable objects to connect to each other — this is what allows us to “chain” components together.  
A Runnable can also be executed using the `.invoke()` method.

This is the power of LangChain: it allows us to seamlessly integrate various components needed to build workflows.  
Everything follows the same interface and the same rules, which makes it easy to connect components and simplifies the development process.  
Additionally, Runnables support different execution patterns: parallel, sequential, and conditional workflows.

Now we can look at how a **chain** works:

$$
\text{chain} = \text{Runnable}_1 \;|\; \text{Runnable}_2 \;|\; \dots \;|\; \text{Runnable}_N
$$

The `|` operator takes the output of the previous Runnable and passes it as the input to the next one.

Now let’s code our final step using chains!


```python
def get_rag_answer(query: str , vector_store: Chroma ):
    load_dotenv()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature = 0.5,
        max_retries = 2
    )
    docs_retriever = RunnableLambda(lambda query: vector_store.similarity_search(query=query,k=6))
    docs_to_text = RunnableLambda(lambda docs: "\n\n".join(doc.page_content for doc in docs))
    query_passthrough = RunnablePassthrough()
    prompt_template = ChatPromptTemplate.from_template(
        """Use the following context to answer the question at the end. 
           You must be respectful and helpful, and answer in the language of the question.
           If you don't know the answer, say that you don't know.
           Context: {context}
           Question: {question}"""
    )
    prompt_runnable = RunnableLambda(lambda args: prompt_template.format_messages(context=args["context"], question=args["query"]))
    pipeline = (
        {
            "context": docs_retriever|docs_to_text,
            "query": query_passthrough
        }
        | prompt_runnable
        | llm
        | StrOutputParser()
    )
    answer = pipeline.invoke(query)
    return answer
```


Everything is working now it's time to put together all of the functions that we wrote !


```python
def ask_question(query:str):
    collection_name = "tcp"
    path = "./pdf_documents"
    persitent_dir = './chromaDB'
    chunks = split_document(path)
    vector_store = store_embeddings_with_langchain(
        collection_name,
        persitent_dir,
        chunks
    )
    answer = get_rag_answer(query,vector_store)
    print(answer) 
    answer = ask_question("Comment fonctionne TCP?")
```


```python
("D'après le contexte fourni :\n"
 '\n'
 "Le protocole TCP assure la transmission des contenus et permet de s'assurer "
 "qu'un paquet... (la phrase est coupée). Il est dit « fiable ».")
```

And here we are the RAG is working ! 

This is the end of this post , I hope you enjoyed It and learned something new .   
Thank you for reading it  and as usual :

$$ 
Keep\;Grinding !
$$
