import requests
import json
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go

# --- 1. DEFINE YOUR SENTENCES ---
# Let's use three distinct topics: technology, local travel, and food.
sentences = [
    "What is quantum computing?",
    "What is quantum computing?",
    "How do large language models work?",
    "The future of artificial intelligence.",
    "Best places to visit in Assam.",
    "Tourist attractions near Guwahati.",
    "What is the weather like in Shillong?",
    "How to cook chicken biryani?",
    "Recipe for authentic masala chai."
]

model_name = "embeddinggemma:latest"


# --- 2. FUNCTION TO GET EMBEDDINGS FROM OLLAMA ---
def get_embedding(prompt, model):
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": prompt}
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API for prompt '{prompt}': {e}")
        return None


print("Generating embeddings for each sentence...")
embeddings = [get_embedding(s, model_name) for s in sentences]

# Filter out any sentences that failed
valid_embeddings = [e for e in embeddings if e is not None]
valid_sentences = [s for s, e in zip(sentences, embeddings) if e is not None]

if len(valid_embeddings) < 2:
    print("Could not generate enough embeddings to create a plot. Exiting.")
else:
    # --- 3. USE T-SNE FOR DIMENSIONALITY REDUCTION ---
    print("Running t-SNE to reduce dimensionality to 2D...")
    X = np.array(valid_embeddings)

    # Perplexity should be less than the number of samples.
    perplexity_value = min(30, len(valid_sentences) - 1)

    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42, init='random', learning_rate=200)
    X_2d = tsne.fit_transform(X)

    # --- 4. CREATE THE INTERACTIVE PLOT ---
    print("Creating the plot...")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            mode='markers+text',
            marker=dict(size=10),
            text=[f"  {s}" for s in valid_sentences],
            textposition="top right",
            hoverinfo='text',
            hovertext=valid_sentences
        )
    )
    fig.update_layout(
        title=f"2D Visualization of '{model_name}' Embeddings",
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2"
    )

    fig.show()
