from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from app.classifier import classify_batch
from fastapi import Header, HTTPException, Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
import os


class ClassifyRequest(BaseModel):
	pixels: list[list[int]]

class ClassifyResponse(BaseModel):
	prediction: str
	confidence: float
	scores: dict[str, float]

app = FastAPI()

@app.get("/health")
def health():
	return {"status": "ok", "model_version": "v1"}

@app.get("/results")
def results():
	return {"results": [], "note": "persistence not yet implemented"}


limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

def verify_api_key(x_api_key: str = Header(...)):
	if x_api_key != os.getenv("SECRET_API_KEY"):
		raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/classify", response_model=ClassifyResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
def classify(request:Request, req: ClassifyRequest):
	arr = np.array(req.pixels, dtype=np.uint8)[np.newaxis]
	return classify_batch(arr)[0]
