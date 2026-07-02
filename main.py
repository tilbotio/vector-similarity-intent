import os

from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer, SimilarityFunction
from typing import Annotated
from urllib.parse import unquote

modelname = 'NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers'

if 'MODELNAME' in os.environ:
    modelname = os.environ['MODELNAME']

model = SentenceTransformer(modelname,
                            similarity_fn_name=SimilarityFunction.COSINE)

app = FastAPI()


@app.get("/")
async def get_intent(user_input: str, intent_options: Annotated[list[str] | None, Query()] = None):
    sentences1 = [unquote(user_input)]

    intent_options_decoded = []
    for io in intent_options:
        intent_options_decoded.append(unquote(io))

    sentences2 = intent_options_decoded

    embeddings1 = model.encode(sentences1)
    embeddings2 = model.encode(sentences2)

    similarities = model.similarity(embeddings1, embeddings2)
    closest = [None, 0.0]
    for idx, sentence2 in enumerate(sentences2):
        print(f" - {sentence2: <30}: {similarities[0][idx]:.4f}")
        if similarities[0][idx] > closest[1]:
            closest = [sentence2, similarities[0][idx]]

    return {"intent": closest[0], "connector_label": closest[0]}

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
