# 2025/12/2
# zhangzhong
# https://milvus.io/docs/quickstart.md
# In this guide we use Milvus Lite, a python library included in pymilvus that can be embedded into the client application

from pymilvus import MilvusClient, model

## Set Up Vector Database
# To create a local Milvus vector database, simply instantiate a MilvusClient by specifying a file name to store all data, such as "milvus_demo.db".
client = MilvusClient("milvus_demo.db")

## Create a Collection
# In Milvus, we need a collection to store vectors and their associated metadata. You can think of it as a table in traditional SQL databases
# When creating a collection, you can define schema and index params to configure vector specs such as dimensionality, index types and distant metrics.
collection_name = "demo_collection"
if client.has_collection(collection_name):
    client.drop_collection(collection_name)
# The primary key and vector fields use their default names (“id” and “vector”).
# The primary key field accepts integers and does not automatically increments (namely not using auto-id feature)
# The metric type (vector distance definition) is set to its default value (COSINE).
# for concrete schema creation: https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/Collections/create_schema.md
client.create_collection(collection_name, dimension=768)


## Prepare Data

# If connection to https://huggingface.co/ failed, uncomment the following path
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# This will download a small embedding model "paraphrase-albert-small-v2" (~50MB).
embedding_fn = model.DefaultEmbeddingFunction()

# Text strings to search from.
# Milvus expects data to be inserted organized as a list of dictionaries, where each dictionary represents a data record, termed as an entity.
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

vectors = embedding_fn.encode_documents(docs)
# The output vector has 768 dimensions, matching the collection that we just created.
print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

# Each entity has id, vector representation, raw text, and a subject label that we use
# to demo metadata filtering later.
data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(vectors))
]

print("Data has", len(data), "entities, each with fields: ", data[0].keys())
print("Vector dim:", len(data[0]["vector"]))


## Insert Data
res = client.insert(collection_name, data=data)

print(res)


## Semantic Search
query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])
# If you don't have the embedding function you can use a fake vector to finish the demo:
# query_vectors = [ [ random.uniform(-1, 1) for _ in range(768) ] ]

res = client.search(
    collection_name,  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)

# The output is a list of results, each mapping to a vector search query
# Each query contains a list of results, where each result contains the entity primary key, the distance to the query vector, and the entity details with specified `output_fields`.
print(res)

## Vector Search with Metadata Filtering
# You can also conduct vector search while considering the values of the metadata (called “scalar” fields in Milvus, as scalar refers to non-vector data).
# This is done with a filter expression specifying certain criteria.


# Insert more docs in another subject.
docs = [
    "Machine learning has been used for drug design.",
    "Computational synthesis with AI algorithms predicts molecular properties.",
    "DDR1 is involved in cancers and fibrosis.",
]
vectors = embedding_fn.encode_documents(docs)
data = [
    {"id": 3 + i, "vector": vectors[i], "text": docs[i], "subject": "biology"}
    for i in range(len(vectors))
]
client.insert(collection_name="demo_collection", data=data)

# This will exclude any text in "history" subject despite close to the query vector.
# ！！！By default, the scalar fields are not indexed. If you need to perform metadata filtered search in large dataset, you can consider using fixed schema and also turn on the index to improve the search performance.
res = client.search(
    collection_name="demo_collection",
    data=embedding_fn.encode_queries(["tell me AI related information"]),
    filter="subject == 'biology'",  # 好丑陋的filter方式
    limit=2,
    output_fields=["text", "subject"],
)

print(res)


## Delete Entities
# Delete entities by primary key
res = client.delete(collection_name="demo_collection", ids=[0, 2])

print(res)

# Delete entities by a filter expression
res = client.delete(
    collection_name="demo_collection",
    filter="subject == 'biology'",
)

print(res)
