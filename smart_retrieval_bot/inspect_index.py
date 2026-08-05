"""Quick sanity check that the FAISS index loads and returns chunks."""

from langchain_community.vectorstores import FAISS

from rag_config import FAISS_INDEX_DIR, get_embeddings

embeddings = get_embeddings()
db = FAISS.load_local(
    FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
)

docs = db.similarity_search("billing insurance claim", k=5)
for i, doc in enumerate(docs):
    print(f"\nChunk {i + 1}:")
    print("Text:", doc.page_content[:180], "...")
    print("Metadata:", doc.metadata)
