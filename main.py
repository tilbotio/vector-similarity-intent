import json
import multiprocessing
import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from sentence_transformers import SentenceTransformer, SimilarityFunction
from uvicorn import run

load_dotenv()

PORT = os.getenv("PORT", 8081)
MODELNAME = os.getenv("MODELNAME", "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers")

model = SentenceTransformer(MODELNAME,
                            similarity_fn_name=SimilarityFunction.COSINE)

app = FastAPI()

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

@app.post("/")
async def get_intent(image: UploadFile = File(None),
    user_input: str = Form(""),
    prompt: str | None = Form(None),
    intent_options: str | None = Form(None),
    key_user_account: str | None = Form(None),
):
    sentences1 = [user_input]
    intent_options_parsed = json.loads(intent_options) if intent_options else []

    sentences2 = [option for option in intent_options_parsed if option != "candidates"]

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
