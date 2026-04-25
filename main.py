from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def get_intent():
    return {"intent": None}

# from sentence_transformers import SentenceTransformer, SimilarityFunction
# sentences1 = ["Draagt de persoon een corrigerend ding gezichts?"]
# sentences2 = ["bril", "ogen", "haar", "leeftijd"]

# # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
# #                             similarity_fn_name=SimilarityFunction.DOT_PRODUCT)
# model = SentenceTransformer('NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers',
#                             similarity_fn_name=SimilarityFunction.COSINE)
# embeddings1 = model.encode(sentences1)
# embeddings2 = model.encode(sentences2)

# similarities = model.similarity(embeddings1, embeddings2)

# for idx_i, sentence1 in enumerate(sentences1):
#     print(sentence1)
#     for idx_j, sentence2 in enumerate(sentences2):
#         print(f" - {sentence2: <30}: {similarities[idx_i][idx_j]:.4f}")
