from fastapi import FastAPI, UploadFile, File
from src.pipelines.inference_pipeline import load_prediction_model, predict_image
import uvicorn

app = FastAPI()

# Global model instance
model, classes = None, None

@app.on_event("startup")
def startup_event():
    global model, classes
    model, classes = load_prediction_model('model_checkpoint.pth')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    label, confidence = predict_image(model, classes, contents)
    return {"class": label, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
