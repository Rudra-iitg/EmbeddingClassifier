 EmbeddingClassifier — Semantic Nexus

Beautiful, local-first embedding exploration with a path to zero-shot image classification.

![banner](https://img.shields.io/badge/Python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-%F0%9F%8C%90-red) ![Ollama](https://img.shields.io/badge/Ollama-local--LLMs-black) ![Embedding Gemma](https://img.shields.io/badge/Model-embeddinggemma%3Alatest-emerald)

## Overview

This project turns natural language into high‑dimensional vectors using the Embedding Gemma model running locally via Ollama, then visualizes and explores them interactively.

- Visual t‑SNE map of sentence embeddings
- Similarity matrix with cosine similarity
- Add/remove data points on the fly
- Retro CRT theme (custom `style.css`)

While the current app focuses on text embeddings, the same embedding workflow can power image classification in a zero‑shot fashion. See “Extend to Image Classification” below for two practical approaches that still let you leverage Embedding Gemma.

## Project structure

```
.
├── main.py                # Quick demo: compute embeddings and plot t‑SNE with Plotly
├── streamlit_app.py       # Full interactive dashboard (Semantic Nexus)
├── style.css              # Retro terminal theme for the app
└── EmbeddingClassifier/
		├── LICENSE
		└── README.md          # You are here
```

## How it works

1. Your text is sent to a local Ollama server exposing the embeddings API.
2. The model `embeddinggemma:latest` returns a dense vector for each input.
3. We reduce vectors to 2D with t‑SNE for visualization and compute cosine similarity for ranking.
4. Streamlit renders an interactive map and a similarity matrix you can explore.

Key files to skim:

- `main.py` — minimal script showing how to call the Ollama embeddings endpoint and plot in 2D.
- `streamlit_app.py` — a richer UI with metrics, t‑SNE map, and similarity matrix; uses `style.css`.

## Prerequisites

- macOS, Linux, or Windows
- Python 3.9+
- Ollama installed and running locally
	- Install: https://ollama.com/download
	- Pull the embedding model: `embeddinggemma:latest`

## Setup (macOS / zsh)

You can use a virtual environment if you prefer.

```bash
# From the project root
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install streamlit requests numpy pandas scikit-learn plotly

# Ensure Ollama is installed, then pull the model once
ollama pull embeddinggemma:latest
# Start the Ollama server if not already running (mac app usually runs it automatically)
ollama serve
```

## Run

Two ways to try it:

1) Minimal demo (static 2D plot)

```bash
python main.py
```

2) Interactive dashboard (recommended)

```bash
streamlit run streamlit_app.py
```

In the app:

- Add sentences in the sidebar and click “Analyze & Plot”
- Explore the Semantic Map (t‑SNE) and the Similarity Matrix
- Remove selected sentences from the dataset as needed

## Configuration

The model name is defined in both scripts as `embeddinggemma:latest`. If you want to switch models or a custom tag, update the constant:

- `main.py` — `model_name = "embeddinggemma:latest"`
- `streamlit_app.py` — `MODEL_NAME = "embeddinggemma:latest"`

If your Ollama server runs on a different host or port, change the API URL accordingly (default is `http://localhost:11434`).

## Troubleshooting

- Ollama connection error: Make sure the server is up and the model is pulled.
- t‑SNE errors with small datasets: Perplexity must be less than the number of samples; the app adjusts this automatically.
- No points on the map: Ensure at least two valid embeddings (e.g., two sentences).

## Extend to Image Classification (Zero‑Shot)

Embedding Gemma is a text embedding model, so there are two reliable paths to leverage it for image classification:

1) Caption → Embed (keeps Gemma for the core embedding)
	 - Use a small vision‑language model (e.g., LLaVA via Ollama) to generate a short caption for the image.
	 - Compute embeddings with Embedding Gemma for:
		 - the generated caption, and
		 - each class label (and optionally a few descriptive prompts per class).
	 - Assign the image to the class with the highest cosine similarity between the caption embedding and the class prompt embedding(s).

	 Pseudocode sketch:

	 ```python
	 # 1) Generate caption with a VLM (e.g., LLaVA served by Ollama)
	 caption = generate_caption_with_llava(image_path)

	 # 2) Embed caption and class prompts with Embedding Gemma
	 cap_vec = embed_with_gemma(caption)
	 class_prompts = {
			 "cat": ["a photo of a cat", "domestic feline animal"],
			 "dog": ["a photo of a dog", "domestic canine animal"],
			 # ...
	 }
	 class_vecs = {c: [embed_with_gemma(p) for p in prompts] for c, prompts in class_prompts.items()}

	 # 3) Score by max cosine similarity across prompts per class
	 def score(vec, vecs):
			 return max(cosine_similarity([vec], [v])[0][0] for v in vecs)

	 best_class = max(class_vecs.keys(), key=lambda c: score(cap_vec, class_vecs[c]))
	 ```

2) Vision Embeddings → Text Labels (best accuracy for purely visual tasks)
	 - Use a vision embedding model (e.g., CLIP/OpenCLIP or `nomic-embed-vision`).
	 - Compute image embeddings and compare them with text label embeddings in the CLIP space (zero‑shot classification).
	 - You can still keep Embedding Gemma for textual analytics in the app (e.g., for captions, metadata, or search), while using CLIP‑space for the vision classifier.

Pick the approach that matches your constraints: if keeping Gemma is a must, use caption→embed; if you need stronger vision performance, use a vision embedding model for the classifier.

## Roadmap

- [ ] Optional image upload + caption pipeline (LLaVA) integrated into the Streamlit UI
- [ ] Optional CLIP/vision‑embedding backend for image classification
- [ ] Export/import datasets
- [ ] Persistent storage (e.g., SQLite) for labeled items

## License

This project is licensed under the terms of the LICENSE file included in this repository. Check the individual licenses for any models you use (Gemma, LLaVA, CLIP, etc.).

## Acknowledgements

- Embedding Gemma (via Ollama)
- Streamlit, Plotly, scikit‑learn