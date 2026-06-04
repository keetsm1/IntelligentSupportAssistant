import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
class Search():
    def __init__(self,query):
        self.query = query
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.prefix = "Represent this sentence for searching relevant passages: " #according to documentation online, 
        #prompting this before the similarity search yields better results

    def conduct_semantic_search(self):

        # converting the query to embeddings
        query_embeddings = self.model.encode(self.query, prompt = self.prefix, convert_to_tensor=True)

        #loading embeddings.npy and answers.npy
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        embeddings_dir = os.path.join(base_dir, "embeddings")
        
        embeddings_path = os.path.join(embeddings_dir, "embeddings.npy")
        answers_path = os.path.join(embeddings_dir, "answers.npy")


        knowledge_base_embeddings = np.load(embeddings_path)
        answers = np.load(answers_path, allow_pickle=True)

        #i was using some CPU to do the similiarity score but needed to GPU version before hand and according to my research,
        #torch allows me to compare data from CPU and data from GPU, not too sure how it technically works but solved issue
        knowledge_base_tensor = torch.tensor(knowledge_base_embeddings, device=query_embeddings.device)
        #cosine similarity
        similarity_scores = util.cos_sim(query_embeddings, knowledge_base_tensor)

        best_match_id = torch.argmax(similarity_scores, dim=1).item()

        return {
            "index": best_match_id,
            "answer": answers[best_match_id],
            "score": float(similarity_scores[0, best_match_id])

        }


