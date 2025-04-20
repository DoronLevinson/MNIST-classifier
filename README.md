# 🧠 MNIST Classifier

A lightweight ML web app that predicts handwritten digits using a neural network model.
Built with:

- **FastAPI** – backend API
- **Streamlit** – frontend UI
- **🐳 Docker & Docker Compose** – for containerized deployment

## 📁 Project Structure
simple_mnist/
├── app/                      # FastAPI backend
│   ├── main.py               # API endpoints and model loading
│   ├── inference.py          # S3 + local utils (download, load, preprocess)
│   ├── model/
│   │   ├── architectures/    # Model architecture (SimpleNN)
│   │   └── weights/          # Downloaded model weights
│   ├── requirements.txt      # FastAPI dependencies
│   ├── Dockerfile
│   └── .dockerignore
├── streamlit/                # Streamlit frontend
│   ├── app.py
│   ├── requirements.txt      # Streamlit dependencies
│   ├── Dockerfile
│   └── .dockerignore
├── docker-compose.yml        # Multi-container setup
├── .env                      # AWS credentials and env variables
└── .gitignore

## 🚀 Getting Started

### Build and run the app

```bash
docker-compose up --build
```

This launches:

✅ FastAPI backend http://localhost:8000/docs (model prediction on /predict)

✅ Streamlit UI on http://localhost:8501

✅ MLflow UI on http://localhost:5000 