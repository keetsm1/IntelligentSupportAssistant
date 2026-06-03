from sentence_transformers import SentenceTransformer

class Search():
    def __init__(self,query):
        self.query = query
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def conduct_semantic_search(self):

        # converting the query to embeddings
        query_embeddings = self.model.encode(self.query)

        
