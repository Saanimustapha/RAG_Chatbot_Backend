from pinecone import Pinecone, ServerlessSpec
from RAG_Chatbot_Backend.core.config import settings

pc = Pinecone(api_key=settings.PINECONE_API_KEY)

def get_or_create_index():
    existing = {i["name"] for i in pc.list_indexes()}
    if settings.PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=settings.PINECONE_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.PINECONE_CLOUD, region=settings.PINECONE_REGION),
        )
    return pc.Index(settings.PINECONE_INDEX_NAME)


def namespace_for_user(user_id: str) -> str:
    # simple isolation per user (Phase 1). Later you can do org/team namespaces.
    return f"{settings.PINECONE_NAMESPACE_PREFIX}{user_id}"

def upsert_vectors(user_id: str, vectors: list[tuple[str, list[float], dict]]):
    """
    vectors: [(id, embedding, metadata)]
    """
    
    index = get_or_create_index()
    index.upsert(vectors=vectors, namespace=namespace_for_user(user_id))

def query_vectors(user_id: str, embedding: list[float], top_k: int):
    
    index = get_or_create_index()
    return index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace_for_user(user_id),
    )
