import streamlit as st
import requests
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from pathlib import Path

# --- App Configuration ---
st.set_page_config(
    page_title="Semantic Nexus",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Inject Custom CSS (relative to this file) ---
this_dir = Path(__file__).resolve().parent
css_path = this_dir / "style.css"
try:
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("style.css file not found. Please ensure it's in the EmbeddingClassifier folder.")

# --- Model Name ---
MODEL_NAME = "embeddinggemma:latest"


# --- Function to Get Embeddings (cached for performance) ---
@st.cache_data(show_spinner=False)
def get_embedding(prompt):
    """Sends a prompt to the Ollama API and returns the embedding."""
    try:
        response = requests.post("http://localhost:11434/api/embeddings", json={"model": MODEL_NAME, "prompt": prompt})
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.RequestException:
        st.error(f"Ollama API Error. Is the Ollama server running?", icon="🛰️")
        return None


# --- Session State Initialization ---
if 'sentences' not in st.session_state:
    st.session_state.sentences = [
        "What is quantum computing?", "How do large language models work?", "The future of artificial intelligence.",
        "Best places to visit in Assam.", "Tourist attractions near Guwahati.", "What is the weather like in Shillong?",
        "How to cook chicken biryani?", "Recipe for authentic masala chai."
    ]

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h1>Semantic Nexus</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Add Data Point")
    new_sentence = st.text_area("Enter text to embed and analyze:", key="new_sentence_input", height=100)
    if st.button("Analyze & Plot", use_container_width=True):
        if new_sentence and new_sentence.strip() not in st.session_state.sentences:
            st.session_state.sentences.append(new_sentence.strip())
            st.rerun()

    st.markdown("---")
    st.subheader("Manage Data Points")
    sentences_to_remove = st.multiselect("Select sentences to remove:", options=st.session_state.sentences)
    if st.button("Remove Selected", use_container_width=True):
        st.session_state.sentences = [s for s in st.session_state.sentences if s not in sentences_to_remove]
        st.rerun()

# --- Main App Body ---
st.markdown("## 🧠 Embedding Analysis Dashboard")

# --- NEW HERO SECTION ---
with st.container():
    st.markdown("#### Welcome to the Semantic Nexus")
    st.write(
        """
        This dashboard is a real-time visualization tool for exploring text embeddings. 
        It uses a locally-run AI model to convert natural language into high-dimensional vectors. 
        The **Semantic Map** below projects these complex vectors into a 2D space, allowing you to visually 
        discover how the model groups sentences based on their conceptual meaning. Use the **Similarity Matrix**
        to get a precise numerical score of how related any two sentences are.
        """
    )
st.markdown("---")  # Visual separator

sentences = st.session_state.sentences

if len(sentences) < 2:
    st.warning("Please add at least two sentences to begin analysis.")
else:
    embeddings_list = [get_embedding(s) for s in sentences]
    valid_embeddings = np.array([e for e in embeddings_list if e is not None])
    valid_sentences = [s for s, e in zip(sentences, embeddings_list) if e is not None]

    if len(valid_embeddings) >= 2:
        # --- Top Metrics ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Data Points", f"{len(valid_sentences)}", "Sentences")
        col2.metric("Vector Dimensions", f"{valid_embeddings.shape[1]}", "per point")
        col3.metric("Model", MODEL_NAME)
        st.markdown("---")

        # --- Main Layout ---
        map_col, analysis_col = st.columns([0.6, 0.4])

        with map_col:
            st.markdown("### 🗺️ Semantic Map (t-SNE)")
            with st.spinner("Calculating 2D projection..."):
                perplexity_value = min(5, len(valid_sentences) - 1)
                if perplexity_value < 1: perplexity_value = 1
                tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42, init='pca',
                            learning_rate='auto', max_iter=1000)
                X_2d = tsne.fit_transform(valid_embeddings)

                import pandas as pd
                df_plot = pd.DataFrame(X_2d, columns=['x', 'y'])
                df_plot['sentence'] = valid_sentences

                fig = px.scatter(
                    df_plot, x='x', y='y', text='sentence',
                    hover_data={'sentence': True, 'x': False, 'y': False}, height=550
                )
                fig.update_traces(textposition='top center', textfont=dict(size=12, color='white'))
                fig.update_layout(
                    title_text='Relative Semantic Positions',
                    xaxis_title=None, yaxis_title=None,
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='var(--text-color)',
                    # --- GRIDLINES MADE VISIBLE HERE ---
                    xaxis=dict(showgrid=True, gridcolor='rgba(100, 100, 100, 0.3)', zeroline=False,
                               showticklabels=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(100, 100, 100, 0.3)', zeroline=False,
                               showticklabels=False),
                    hoverlabel=dict(bgcolor="var(--bg-color)", font_size=12, font_family="Roboto")
                )
                st.plotly_chart(fig, use_container_width=True)

        with analysis_col:
            st.markdown("### 📊 Similarity Matrix")
            selected_sentence = st.selectbox("Select a reference sentence:", options=valid_sentences)

            if selected_sentence:
                selected_idx = valid_sentences.index(selected_sentence)
                selected_embedding = valid_embeddings[selected_idx].reshape(1, -1)

                similarity_scores = cosine_similarity(selected_embedding, valid_embeddings)[0]

                import pandas as pd
                df_similarity = pd.DataFrame({"Similarity": similarity_scores, "Sentence": valid_sentences})
                df_similarity = df_similarity.sort_values(by="Similarity", ascending=False).reset_index(drop=True)

                st.dataframe(
                    df_similarity,
                    use_container_width=True,
                    column_config={
                        "Similarity": st.column_config.ProgressColumn("Similarity", format="%.3f", min_value=0,
                                                                      max_value=1)}
                )
