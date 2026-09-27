import multiprocessing
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, SimilarityFunction
from uvicorn import run

load_dotenv()

PORT = os.getenv("PORT", 8081)
MODELNAME = os.getenv("MODELNAME", "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers")

model = SentenceTransformer(MODELNAME,
                            similarity_fn_name=SimilarityFunction.COSINE)

app = FastAPI()


class IntentRequest(BaseModel):
    user_input: str = ""
    intent_options: list[str] = []


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

@app.post("/")
async def get_intent(request: IntentRequest):
    sentences1 = [request.user_input]

    sentences2 = [option for option in request.intent_options if option != "candidates"]

    embeddings1 = model.encode(sentences1)
    embeddings2 = model.encode(sentences2)

    similarities = model.similarity(embeddings1, embeddings2)
    closest = [None, 0.0]
    secondclosest = [None, 0.0]
    thirdclosest = [None, 0.0]

    for idx, sentence2 in enumerate(sentences2):
        print(f" - {sentence2: <30}: {similarities[0][idx]:.4f}")
        if similarities[0][idx] > closest[1]:
            thirdclosest = secondclosest
            secondclosest = closest
            closest = [sentence2, similarities[0][idx]]
        elif similarities[0][idx] > secondclosest[1]:
            secondclosest = [sentence2, similarities[0][idx]]
        elif similarities[0][idx] > thirdclosest[1]:
            thirdclosest = [sentence2, similarities[0][idx]]

    if closest[1] > 0.4:
        if closest[1] - secondclosest[1] < 0.05:
            if secondclosest[1] - thirdclosest[1] < 0.05:
                return {"intent": "candidates", "connector_label": [closest[0], secondclosest[0], thirdclosest[0]]}
            return {"intent": "candidates", "connector_label": [closest[0], secondclosest[0]]}

        return {"intent": closest[0], "connector_label": closest[0]}
    else:
        return None

if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows support
    run(app, host="0.0.0.0", port=PORT, reload=False, workers=1)
