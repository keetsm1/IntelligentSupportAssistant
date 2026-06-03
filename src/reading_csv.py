import os
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
knowledge_csv = pd.read_csv(os.path.join(base_dir, 'knowledge_base.csv'), skipinitialspace=True)

questions = knowledge_csv['question']
answer = knowledge_csv['answer']
