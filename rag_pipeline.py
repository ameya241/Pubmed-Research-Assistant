import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from fetch_pubmed import fetch_papers


class MedicalRAG:
    def __init__(self):
        # Embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # ✅ Use text-generation instead (stable)
        self.generator = pipeline(
            "text-generation",
            model="distilgpt2",
        )

        self.index = None
        self.documents = []

    def load_data(self, query="diabetes treatment"):
        papers = fetch_papers(query)
        chunks = papers.split("\n\n")

        self.documents = [chunk for chunk in chunks if len(chunk) > 100]

        embeddings = self.embedding_model.encode(self.documents)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def ask(self, question, k=3):
        question_embedding = self.embedding_model.encode([question])
        question_embedding = np.array(question_embedding).astype("float32")

        distances, indices = self.index.search(question_embedding, k)

        context = ""
        citations = []

        for idx in indices[0]:
            context += self.documents[idx] + "\n"
            citations.append(self.documents[idx][:200])

        prompt = f"""
        Use the following medical context to answer the question clearly.

        Context:
        {context}

        Question: {question}

        Answer:
        """

        response = self.generator(
            prompt,
            max_length=300,
            do_sample=False
        )

        answer = response[0]["generated_text"]

        return answer, citations
