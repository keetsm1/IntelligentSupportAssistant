import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
knowledge_csv = pd.read_csv(os.path.join(base_dir, 'knowledge_base.csv'), skipinitialspace=True)


#fills missing vals, converts the rows into strings and extracts as a list
questions = knowledge_csv['question'].fillna('').astype(str).tolist()
answers = knowledge_csv['answer'].fillna('').astype(str).tolist()


#choose model for generating embeddings.
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

embeddings  = model.encode(questions)

#being saved in numpy file because faster + better for embedding. Idea:
# user types question --> we convert it into embedding
# compare with stored embeddings using semantic search
# we get the index of the closest match
# we use that index to retrieve the corresponding answer
embeddings_dir = os.path.join(base_dir, "embeddings")


np.save(os.path.join(embeddings_dir, "questions.npy"), questions)
np.save(os.path.join(embeddings_dir, "answers.npy"), answers)
np.save(os.path.join(embeddings_dir, "embeddings.npy"), embeddings)