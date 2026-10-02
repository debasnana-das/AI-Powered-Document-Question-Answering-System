import numpy as np
from src.document_processor import DocumentChunk
from src.vector_store import VectorStore

def test_vector_store_retrieval():
    chunks=[DocumentChunk("work from home up to three days","policy.pdf",1,1),DocumentChunk("annual leave is twenty days","policy.pdf",2,2)]
    vectors=np.asarray([[1.0,0.0],[0.0,1.0]],dtype=np.float32)
    store=VectorStore(); store.add(chunks,vectors)
    results=store.search(np.asarray([1.0,0.0],dtype=np.float32),top_k=1)
    assert results[0][0].text.startswith("work from home")
