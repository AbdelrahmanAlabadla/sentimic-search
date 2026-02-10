import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import seaborn as sns

class SematicSearchEngine:
    def __init__(self,csv_path,model_name):
        self.df=pd.read_csv(csv_path)
        self.sentences=self.df["Sentence"].tolist()

        self.model=SentenceTransformer(model_name)
        self.embedding=None
    def encode_sentences(self):
        self.embeddings=normalize(self.model.encode(self.sentences
                                         ,convert_to_tensor=True))
        print("Embeddings shape:", self.embeddings.shape)

    def search(self,query,top_k=5):
        query_embedding= normalize(self.model.encode(query,convert_to_tensor=True).reshape(1,-1).cpu().numpy())
        similarity_matrix=cosine_similarity(query_embedding,
                                            self.embeddings)[0]

        top_search=similarity_matrix.argsort()[::-1][:top_k]

        print("\nSimilarity results:")

        for idx in top_search:
            score=similarity_matrix[idx]
            if score < 0.6:
                continue
            category=self.df.loc[idx,"Category"]
            sentence=self.df.loc[idx,"Sentence"]



            print(f"[{category}] "
                  f"{sentence} - Score : {score:.4f}")

        else:
            print("there is now simialry ")



engine=SematicSearchEngine(
    csv_path="Sentimic_Search.csv",
    model_name="all-MiniLM-L6-v2"
)

engine.encode_sentences()
engine.search("Why is   my payment     still pending even    though the money was taken from my bank?")
