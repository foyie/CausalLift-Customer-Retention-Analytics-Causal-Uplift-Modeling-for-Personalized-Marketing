from fastapi import FastAPI
import pickle

app = FastAPI()

# Load model
with open('models/causal_forest.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(features: dict):
    # Preprocess and predict CATE
    cate = model.predict(features)
    return {"cate": float(cate)}
