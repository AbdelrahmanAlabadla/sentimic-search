import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

class SearchSemanticEngine:
    def __init__(self,csv_path,model):
        self.df=pd.read_csv(csv_path)
        self.sentences=self.df["Sentence"].tolist()
        self.model=SentenceTransformer(model)
        self.embeddings=None


    def encode_sentences(self):
        self.embeddings=normalize(self.model.encode(self.sentences,convert_to_numpy=True))
        print("embedding shape: ",self.embeddings.shape)

    def preprocess_text(self,text):
        text=" ".join(text.split())
        text=text.strip()
        return text

    def search(self,query,top_k=5):
        query=self.preprocess_text(query)
        query_embeddings=normalize(self.model.encode(query,convert_to_numpy=True).reshape(1,-1))
        similarity_matrix=cosine_similarity(query_embeddings,self.embeddings)[0]

        top_search=similarity_matrix.argsort()[::-1][:top_k]
        print("\nSimilarity results:")


        result=[]
        found = False
        for idx in top_search:
            score=similarity_matrix[idx]
            if score < 0.6:
                continue

            found=True
            category=self.df.loc[idx,"Category"]
            sentence=self.df.loc[idx,"Sentence"]
            print(f"[{category}] {sentence} - Score: {score:.4f}")

            result.append({"sentence": sentence, "category": category, "score": score})
        if not found:
            print("No similar sentences above the threshold")
        return result


engine=SearchSemanticEngine(
    csv_path="Sentimic_Search.csv",
    model="all-MiniLM-L6-v2"
)

engine.encode_sentences()
engine.search("Why is   my payment     still pending even    though the money was taken from my bank?")









