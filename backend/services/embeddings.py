from sentence_transformers import SentenceTransformer
import numpy as np

def doTheThing():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = [
        "This is the first sentence",
        "This is the second sentence",
        "The History and Future of Humanity, written by Aroden",
        "The fitnessgram pacer test is a multistage aerobic capacity test",
        "The Gerald R. Ford-class nuclear-powered aircraft carriers are currently being constructed for the United States Navy",
        "Yamato was the lead ship of her class of battleships built for the imperial Japanese Navy (IJN) shortly before World War II",
        "It is cold outside",
        "It is hot out",
    ]

    print("encoding starting!!")

    embeddings = model.encode(sentences)
    print(embeddings.shape)

    similarities = model.similarity(embeddings, embeddings)
    print(similarities)

if __name__ == "__main__":
    doTheThing()