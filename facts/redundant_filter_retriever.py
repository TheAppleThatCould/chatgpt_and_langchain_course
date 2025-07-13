from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever


class ReduntdantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    # Filer out duplicate in ./emb vector storage
    def get_relevant_documents(self, query):
        # Calculate embeedings for the'query string
        emb = self.embeddings.embed_query(query)

        # take embeddings and feed them into
        # max_marginal_relevance_search_by_vector
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8
        )
    
    async def aget_relevant_documents(self):
        return []