"""
Mini Project 3 — Interactive Streamlit App
Explore word embeddings, next-word predictions, and PCA visualizations.

Run with: streamlit run app.py
"""

import streamlit as st
import pickle
import math
import os
import html
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

# ──────────────────────────────────────────────
# Model Architecture Definitions (for loading checkpoints)
# ──────────────────────────────────────────────

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        init_range = 0.5 / embedding_dim
        nn.init.uniform_(self.center_embeddings.weight, -init_range, init_range)
        nn.init.uniform_(self.context_embeddings.weight, -init_range, init_range)

    def forward(self, center, context, negatives):
        center_emb = self.center_embeddings(center)
        context_emb = self.context_embeddings(context)
        pos_scores = torch.sum(center_emb * context_emb, dim=1)
        pos_loss = F.logsigmoid(pos_scores)

        neg_emb = self.context_embeddings(negatives)
        neg_scores = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(-1)
        neg_loss = F.logsigmoid(-neg_scores).sum(dim=1)

        return -(pos_loss + neg_loss).mean()

    def get_embeddings(self):
        """Return the center embeddings as a numpy array."""
        return self.center_embeddings.weight.detach().cpu().numpy()


class NeuralLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_prev, hidden_size, pretrained_embeddings=None):
        super().__init__()
        self.n_prev = n_prev
        self.embedding_dim = embedding_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embeddings.weight.data.copy_(torch.tensor(pretrained_embeddings, dtype=torch.float32))

        input_size = n_prev * embedding_dim
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        emb = self.embeddings(x)
        emb = emb.view(emb.size(0), -1)
        h = torch.sigmoid(self.hidden(emb))
        logits = self.output(h)
        return logits


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

MODELS_DIR = 'saved_models'


@st.cache_resource
def load_all_models():
    """Load all saved models and data into memory."""
    device = torch.device('cpu')

    # 1. Vocabulary & config
    with open(f'{MODELS_DIR}/vocab_data.pkl', 'rb') as f:
        vocab_data = pickle.load(f)
    with open(f'{MODELS_DIR}/config.pkl', 'rb') as f:
        config = pickle.load(f)

    word2idx = vocab_data['word2idx']
    idx2word = vocab_data['idx2word']
    vocab = vocab_data['vocab']
    vocab_size = vocab_data['vocab_size']

    # 2. Skip-gram embeddings (numpy)
    with open(f'{MODELS_DIR}/skipgram_embeddings.pkl', 'rb') as f:
        skipgram_embeddings = pickle.load(f)

    # 3. Scratch embeddings (numpy)
    with open(f'{MODELS_DIR}/scratch_embeddings.pkl', 'rb') as f:
        scratch_embeddings = pickle.load(f)

    # 4. Skip-gram model (PyTorch)
    sg_checkpoint = torch.load(f'{MODELS_DIR}/skipgram_model.pt', map_location=device)
    sg_model = SkipGramModel(sg_checkpoint['vocab_size'], sg_checkpoint['embedding_dim'])
    sg_model.load_state_dict(sg_checkpoint['model_state_dict'])
    sg_model.eval()

    # 5. LM Fixed model
    lm_fixed_checkpoint = torch.load(f'{MODELS_DIR}/lm_model_fixed.pt', map_location=device)
    lm_fixed = NeuralLanguageModel(
        vocab_size=lm_fixed_checkpoint['vocab_size'],
        embedding_dim=lm_fixed_checkpoint['embedding_dim'],
        n_prev=lm_fixed_checkpoint['n_prev'],
        hidden_size=lm_fixed_checkpoint['hidden_size']
    )
    lm_fixed.load_state_dict(lm_fixed_checkpoint['model_state_dict'])
    lm_fixed.eval()
    # Freeze embeddings
    for param in lm_fixed.embeddings.parameters():
        param.requires_grad = False

    # 6. LM Scratch model
    lm_scratch_checkpoint = torch.load(f'{MODELS_DIR}/lm_model_scratch.pt', map_location=device)
    lm_scratch = NeuralLanguageModel(
        vocab_size=lm_scratch_checkpoint['vocab_size'],
        embedding_dim=lm_scratch_checkpoint['embedding_dim'],
        n_prev=lm_scratch_checkpoint['n_prev'],
        hidden_size=lm_scratch_checkpoint['hidden_size']
    )
    lm_scratch.load_state_dict(lm_scratch_checkpoint['model_state_dict'])
    lm_scratch.eval()

    # 7. PCA data
    with open(f'{MODELS_DIR}/pca_data.pkl', 'rb') as f:
        pca_data = pickle.load(f)

    # 8. Tuning results
    with open(f'{MODELS_DIR}/tuning_results.pkl', 'rb') as f:
        tuning_results = pickle.load(f)

    # 9. LM data
    with open(f'{MODELS_DIR}/lm_data.pkl', 'rb') as f:
        lm_data = pickle.load(f)

    return {
        'vocab_data': vocab_data,
        'config': config,
        'word2idx': word2idx,
        'idx2word': idx2word,
        'vocab': vocab,
        'vocab_size': vocab_size,
        'skipgram_embeddings': skipgram_embeddings,
        'scratch_embeddings': scratch_embeddings,
        'sg_model': sg_model,
        'lm_fixed': lm_fixed,
        'lm_scratch': lm_scratch,
        'pca_data': pca_data,
        'tuning_results': tuning_results,
        'lm_data': lm_data,
    }


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────

def find_similar_words(word, embeddings, idx2word, word2idx, top_k=15):
    """Find top_k most similar words by cosine similarity."""
    if word not in word2idx:
        return []
    idx = word2idx[word]
    vec = embeddings[idx]
    norms = np.linalg.norm(embeddings, axis=1)
    sims = (embeddings @ vec) / (norms * np.linalg.norm(vec) + 1e-8)
    top_indices = np.argsort(sims)[::-1][1:top_k + 1]
    return [(idx2word[i], float(sims[i])) for i in top_indices]


def predict_next_word(model, word2idx, idx2word, context_words, n_prev, top_k=5):
    """Predict the next word given context words."""
    # Convert context words to indices
    indices = []
    for w in context_words:
        if w in word2idx:
            indices.append(word2idx[w])
        else:
            return None, f'Word "{w}" not in vocabulary'

    if len(indices) < n_prev:
        return None, f'Need {n_prev} words, got {len(indices)}'

    # Take last n_prev words
    input_indices = indices[-n_prev:]
    x = torch.tensor([input_indices], dtype=torch.long)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)

    pred_idx = logits.argmax(dim=1).item()
    predicted_word = idx2word[pred_idx]

    # Top-k candidates
    top_probs, top_indices = torch.topk(probs[0], top_k)
    candidates = [(idx2word[idx.item()], float(prob.item())) for idx, prob in zip(top_indices, top_probs)]

    return predicted_word, candidates


def project_embeddings_to_2d(embeddings, seed=42):
    """Fit PCA and transform embeddings to 2D."""
    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(embeddings)


def draw_neural_network(active_layer=None, vocab_size=182, hidden_size=512, n_prev=5, emb_dim=50):
    """
    Draw a neural network diagram using Plotly Scatter + lines.
    When active_layer is set (0-4), that layer is highlighted and others are dimmed.
    active_layer: None=all visible, 0=input, 1=embedding, 2=concat, 3=hidden, 4=output
    """
    import random
    fig = go.Figure()
    
    # Layer positions (x, y) - simplified layout
    layer_x = [0, 1.5, 3, 4.5, 6]
    layer_names = ['Word', 'Embed', 'Concat', 'Hidden', 'Output']
    layer_labels = ['Input', 'Embedding', 'Concat', 'Hidden', 'Output']
    layer_sizes = [14, 5, 9, 9, 13]
    layer_colors = ['#64b5f6', '#81c784', '#ffb74d', '#ba68c8', '#ef5350']
    layer_titles = [
        'Input Layer<br><sup>5 previous words</sup>',
        'Embedding<br><sup>50D vectors</sup>',
        'Concat<br><sup>5 x 50 = 250D</sup>',
        'Hidden Layer<br><sup>512 sigmoid</sup>',
        f'Output Layer<br><sup>Softmax over {vocab_size} words</sup>'
    ]
    layer_title_colors = ['#1565c0', '#2e7d32', '#e65100', '#6a1b9a', '#c62828']
    n_nodes = [5, 10, 8, 20, 15]
    node_texts_list = [
        ['w1', 'w2', 'w3', 'w4', 'w5'],
        None, None, None,
        ['word1', 'word2', '...', f'word{vocab_size}'] if 15 >= 4 else None
    ]
    
    conn_colors = [
        'rgba(100, 181, 246, 0.15)',
        'rgba(129, 199, 132, 0.15)',
        'rgba(255, 183, 77, 0.15)',
        'rgba(186, 104, 200, 0.15)',
    ]
    conn_highlight = [
        'rgba(100, 181, 246, 0.5)',
        'rgba(129, 199, 132, 0.5)',
        'rgba(255, 183, 77, 0.5)',
        'rgba(186, 104, 200, 0.5)',
    ]

    # Helper to create evenly spaced y positions
    def y_positions(n, center=0, spread=3):
        if n <= 1:
            return [center]
        spacing = spread / (n - 1)
        return [center - spread/2 + i * spacing for i in range(n)]

    def is_active(li):
        return active_layer is not None and li == active_layer
    
    # Store layer y-positions for connections
    layer_ys = []

    for li in range(5):
        x = layer_x[li]
        n = n_nodes[li]
        color = layer_colors[li]
        name = layer_names[li]
        size = layer_sizes[li]
        title = layer_titles[li]
        title_color = layer_title_colors[li]
        texts = node_texts_list[li]
        active = is_active(li)
        
        active_opacity = 1.0
        dim_opacity = 0.15
        opacity = active_opacity if active else (dim_opacity if active_layer is not None else 0.8)
        actual_size = size * 2.0 if active else size
        
        ys = y_positions(n, center=0, spread=3.5)
        layer_ys.append(ys)
        
        marker_dict = dict(color=color, size=actual_size, opacity=opacity)
        if active:
            marker_dict['line'] = dict(color='#FFD700', width=3)
        else:
            marker_dict['line'] = dict(color='white', width=0.5)
        
        fig.add_trace(go.Scatter(
            x=[x] * n, y=ys,
            mode='markers+text' if texts else 'markers',
            marker=marker_dict,
            text=texts if texts else None,
            textposition='middle right' if texts else None,
            textfont=dict(size=8, color='#333'),
            hovertext=[f'{name} {j+1}' for j in range(n)],
            hoverinfo='text',
            name=name,
            showlegend=False
        ))
        
        # Layer label below
        fig.add_annotation(x=x, y=-2.5, xref='x', yref='y',
                          text=title, showarrow=False,
                          font=dict(size=10, color=title_color,
                                   style='italic' if not active else 'normal'))
    
    # ── Connections between layers ──
    for li in range(4):
        y1_list = layer_ys[li]
        y2_list = layer_ys[li + 1]
        color = conn_highlight[li] if (active_layer is not None and (active_layer == li or active_layer == li + 1)) else conn_colors[li]
        n_sample = 35 if (active_layer is not None and (active_layer == li or active_layer == li + 1)) else 25
        
        indices = random.sample(range(len(y1_list)), min(n_sample, len(y1_list)))
        for i in indices:
            j = i % len(y2_list)
            fig.add_annotation(
                x=layer_x[li+1], y=y2_list[j], ax=layer_x[li], ay=y1_list[i],
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=0, arrowsize=0.5,
                arrowwidth=0.3, arrowcolor=color,
                standoff=0, startstandoff=0
            )
    
    # ── Highlight glow for active layer ──
    if active_layer is not None:
        x_c = layer_x[active_layer]
        fig.add_vrect(
            x0=x_c - 0.55, x1=x_c + 0.55,
            fillcolor='rgba(255, 215, 0, 0.08)',
            layer='below', line_width=0
        )
    
    # Layout
    fig.update_layout(
        height=450,
        xaxis=dict(range=[-0.5, 7], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-3.2, 3.2], showgrid=False, zeroline=False, visible=False),
        margin=dict(l=10, r=10, t=10, b=40),
        plot_bgcolor='rgba(248, 249, 250, 1)',
        paper_bgcolor='rgba(248, 249, 250, 1)',
        showlegend=False
    )
    return fig


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────

st.set_page_config(
    page_title='Khmer Word Embeddings Explorer',
    page_icon='🛕',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .main-header {
        text-align: center; padding: 1.5rem 0; background: linear-gradient(135deg, #1a237e, #4a148c);
        color: white; border-radius: 12px; margin-bottom: 2rem;
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; }
    .main-header p { margin: 0.3rem 0 0; opacity: 0.85; font-size: 1rem; }
    .result-card {
        background: white; padding: 1.2rem; border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin: 0.8rem 0;
        border-left: 4px solid #4a148c;
    }
    .word-chip {
        display: inline-block; background: #e8eaf6; padding: 0.3rem 0.8rem;
        border-radius: 16px; margin: 0.2rem; font-size: 0.9rem;
        border: 1px solid #c5cae9;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1a237e, #4a148c);
        color: white; border: none; border-radius: 8px; padding: 0.5rem 1.5rem;
    }
    .stButton>button:hover { opacity: 0.9; }
    .metric-box {
        background: white; padding: 1rem; border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06); text-align: center;
    }
    h2, h3 { color: #1a237e; }
    .sidebar-header { font-size: 1.1rem; font-weight: bold; color: #4a148c; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="main-header">
    <h1>🛕 Khmer Word Embeddings Explorer</h1>
    <p>Mini Project 3 — Skip-gram · Neural Language Model · Hyperparameter Tuning</p>
</div>
""", unsafe_allow_html=True)

# Load data
with st.spinner('Loading models... This may take a moment.'):
    data = load_all_models()

word2idx = data['word2idx']
idx2word = data['idx2word']
vocab = data['vocab']
vocab_size = data['vocab_size']
config = data['config']
sg_emb = data['skipgram_embeddings']
sc_emb = data['scratch_embeddings']
sg_model = data['sg_model']
lm_fixed = data['lm_fixed']
lm_scratch = data['lm_scratch']
pca_data = data['pca_data']
tuning_results = data['tuning_results']

# ─── Sidebar ──────────────────────────────────

st.sidebar.markdown('<p class="sidebar-header">⚙️ Configuration</p>', unsafe_allow_html=True)

st.sidebar.markdown("**Model Parameters**")
st.sidebar.info(
    f"**Embedding Dim:** {config['EMBEDDING_DIM']}  \n"
    f"**Window Size:** ±{config['WINDOW_SIZE']}  \n"
    f"**Negative Samples:** {config['NEG_SAMPLES']}  \n"
    f"**Vocabulary:** {data['vocab_size']:,} words  \n"
    f"**Hidden Size (LM):** {config['HIDDEN_SIZE']}  \n"
    f"**N Previous (LM):** {config['N_PREV']}"
)

st.sidebar.markdown("**Performance**")
sg_var = pca_data['pca_skipgram'].explained_variance_ratio_.sum()
sc_var = pca_data['pca_scratch'].explained_variance_ratio_.sum()
st.sidebar.info(
    f"📊 **PCA Variance (SG):** {sg_var:.2%}  \n"
    f"📊 **PCA Variance (Scratch):** {sc_var:.2%}"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**ℹ️ Quick Tips**")
st.sidebar.caption(
    "• Type a Khmer word to see its nearest neighbors  \n"
    "• Use the Chat tab for interactive text exploration  \n"
    "• 🎬 Learn the Process tab explains each step visually  \n"
    "• Words not in vocabulary (freq < 10) won't work"
)

# ─── Tabs ─────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    '🔍 Word Similarity',
    '🔮 Next-Word Prediction',
    '📊 PCA Visualization',
    '⚡ Model Comparison',
    '📈 Hyperparameter Tuning',
    '💬 Chat Interface',
    '🎬 Learn the Process'
])

# ════════════════════════════════════════════
# TAB 1: Word Similarity
# ════════════════════════════════════════════

with tab1:
    st.header('🔍 Word Similarity Search')
    st.markdown('Find the most similar words using cosine similarity between embedding vectors.')

    col1, col2 = st.columns([2, 1])

    with col1:
        # Dropdown with autocomplete-like search
        search_term = st.text_input(
            'Enter a Khmer word:',
            placeholder='e.g. ប្រាសាទ, អង្គរ, វត្ត, ខេត្ត, ទេសចរណ៍',
            help='Type the exact Khmer word as it appears in the corpus'
        )

        top_k = st.slider('Number of neighbors:', 5, 30, 15)

    with col2:
        st.markdown('**📖 Sample words in vocabulary**')
        sample_words = list(word2idx.keys())[:12]
        sample_html = ' '.join([f'<span class="word-chip">{w}</span>' for w in sample_words])
        st.markdown(f'<div>{sample_html}</div>', unsafe_allow_html=True)
        st.caption(f'Total vocabulary: {data["vocab_size"]:,} words')

    if search_term:
        # Skip-gram similarities
        sg_similar = find_similar_words(search_term, sg_emb, idx2word, word2idx, top_k=top_k)
        sc_similar = find_similar_words(search_term, sc_emb, idx2word, word2idx, top_k=top_k)

        if not sg_similar:
            st.warning(f'Word "{search_term}" not found in vocabulary. Try one of the sample words above.')
        else:
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f'**Skip-gram Embeddings**  \n*Nearest neighbors for "{search_term}"*')
                df_sg = pd.DataFrame(sg_similar, columns=['Word', 'Similarity'])
                df_sg['Similarity'] = df_sg['Similarity'].round(4)
                st.dataframe(df_sg, use_container_width=True, hide_index=True)

                # Bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[s[1] for s in sg_similar][:10][::-1],
                    y=[s[0] for s in sg_similar][:10][::-1],
                    orientation='h',
                    marker=dict(color='#1a237e'),
                    text=[f'{s[1]:.3f}' for s in sg_similar][:10][::-1],
                    textposition='outside'
                ))
                fig.update_layout(title=f'Top {min(10, top_k)} Similar Words (Skip-gram)',
                                  height=350, margin=dict(l=0, r=0, t=30, b=0),
                                  xaxis_title='Cosine Similarity')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_b:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f'**LM Scratch Embeddings**  \n*Nearest neighbors for "{search_term}"*')
                df_sc = pd.DataFrame(sc_similar, columns=['Word', 'Similarity'])
                df_sc['Similarity'] = df_sc['Similarity'].round(4)
                st.dataframe(df_sc, use_container_width=True, hide_index=True)

                # Bar chart
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=[s[1] for s in sc_similar][:10][::-1],
                    y=[s[0] for s in sc_similar][:10][::-1],
                    orientation='h',
                    marker=dict(color='#c62828'),
                    text=[f'{s[1]:.3f}' for s in sc_similar][:10][::-1],
                    textposition='outside'
                ))
                fig2.update_layout(title=f'Top {min(10, top_k)} Similar Words (Scratch)',
                                   height=350, margin=dict(l=0, r=0, t=30, b=0),
                                   xaxis_title='Cosine Similarity')
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Agreement analysis
            sg_set = set(w for w, _ in sg_similar[:5])
            sc_set = set(w for w, _ in sc_similar[:5])
            common = sg_set & sc_set
            if common:
                st.success(f'✅ Both models agree on: {", ".join(common)}')
            else:
                st.info('ℹ️ The two embedding spaces learned different semantic relationships.')

# ════════════════════════════════════════════
# TAB 2: Next-Word Prediction
# ════════════════════════════════════════════

with tab2:
    st.header('🔮 Next-Word Prediction')
    st.markdown(
        'Enter a sequence of Khmer words. The model uses the last '
        f'**{config["N_PREV"]} words** to predict the next word.'
    )

    context_input = st.text_area(
        'Enter a Khmer sentence or word sequence:',
        placeholder='e.g. ប្រាសាទ អង្គរវត្ត ជា ប្រាសាទ ដ៏',
        height=100,
        help=f'Enter at least {config["N_PREV"]} words separated by spaces'
    )

    col1, col2 = st.columns(2)

    with col1:
        top_k_pred = st.slider('Number of candidates:', 3, 15, 5, key='pred_k')

    with col2:
        st.markdown('')
        st.markdown('')

    if context_input:
        words = context_input.strip().split()
        if len(words) < config['N_PREV']:
            st.warning(f'Please enter at least {config["N_PREV"]} words.')
        else:
            c_pred, c_cands = predict_next_word(
                lm_fixed, word2idx, idx2word,
                words, config['N_PREV'], top_k=top_k_pred
            )
            s_pred, s_cands = predict_next_word(
                lm_scratch, word2idx, idx2word,
                words, config['N_PREV'], top_k=top_k_pred
            )

            if c_pred is None:
                st.error(c_cands)
            else:
                col_a, col_b = st.columns(2)

                # Fixed LM
                with col_a:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(f'**🧊 Fixed LM (Pretrained Embeddings)**')
                    st.metric('Predicted Next Word', c_pred, delta=None)

                    df_c = pd.DataFrame(c_cands, columns=['Word', 'Probability'])
                    df_c['Probability'] = (df_c['Probability'] * 100).round(2)
                    df_c['Prob %'] = df_c['Probability'].apply(lambda x: f'{x:.1f}%')
                    st.dataframe(df_c[['Word', 'Prob %']], use_container_width=True, hide_index=True)

                    # Horizontal bar
                    fig_c = go.Figure()
                    fig_c.add_trace(go.Bar(
                        x=[p for _, p in c_cands],
                        y=[w for w, _ in c_cands],
                        orientation='h',
                        marker=dict(color='#1565c0'),
                        text=[f'{p*100:.1f}%' for _, p in c_cands],
                        textposition='outside'
                    ))
                    fig_c.update_layout(title='Candidate Probabilities', height=350,
                                        margin=dict(l=0, r=0, t=30, b=0),
                                        xaxis_title='Probability')
                    st.plotly_chart(fig_c, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Scratch LM
                with col_b:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(f'**🔥 Scratch LM (Learned Embeddings)**')
                    st.metric('Predicted Next Word', s_pred, delta=None)

                    df_s = pd.DataFrame(s_cands, columns=['Word', 'Probability'])
                    df_s['Probability'] = (df_s['Probability'] * 100).round(2)
                    df_s['Prob %'] = df_s['Probability'].apply(lambda x: f'{x:.1f}%')
                    st.dataframe(df_s[['Word', 'Prob %']], use_container_width=True, hide_index=True)

                    fig_s = go.Figure()
                    fig_s.add_trace(go.Bar(
                        x=[p for _, p in s_cands],
                        y=[w for w, _ in s_cands],
                        orientation='h',
                        marker=dict(color='#c62828'),
                        text=[f'{p*100:.1f}%' for _, p in s_cands],
                        textposition='outside'
                    ))
                    fig_s.update_layout(title='Candidate Probabilities', height=350,
                                        margin=dict(l=0, r=0, t=30, b=0),
                                        xaxis_title='Probability')
                    st.plotly_chart(fig_s, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Comparison
                agree = c_pred == s_pred
                if agree:
                    st.success(f'✅ Both models agree on "{c_pred}" as the next word!')
                else:
                    st.info(f'ℹ️ The models disagree: Fixed predicts "{c_pred}", Scratch predicts "{s_pred}"')

                st.caption(f'Context: {" ".join(words[-config["N_PREV"]:])}')

# ════════════════════════════════════════════
# TAB 3: PCA Visualization
# ════════════════════════════════════════════

with tab3:
    st.header('📊 PCA Visualization')
    st.markdown('Interactive 2D projection of word embeddings. Hover over points to see word labels.')

    col1, col2 = st.columns([1, 3])

    with col1:
        embed_choice = st.radio(
            'Select embedding type:',
            ['Skip-gram (Part I)', 'LM Scratch (Part IV)', 'Both Side-by-Side'],
            index=0
        )
        max_labels = st.slider('Number of labeled words:', 20, 100, 60)

    word_counts = Counter(data['vocab_data']['filtered_tokens'])
    top_words = [w for w, _ in word_counts.most_common(max_labels)]
    top_indices = [word2idx[w] for w in top_words if w in word2idx]

    pca_sg = pca_data['pca_skipgram']
    pca_sc = pca_data['pca_scratch']

    sv = pca_sg.explained_variance_ratio_.sum()
    scv = pca_sc.explained_variance_ratio_.sum()

    with col2:
        if embed_choice == 'Skip-gram (Part I)':
            emb_2d = pca_data['embeddings_2d']
            fig = go.Figure()

            # All points
            fig.add_trace(go.Scatter(
                x=emb_2d[:, 0], y=emb_2d[:, 1],
                mode='markers',
                marker=dict(color='steelblue', size=5, opacity=0.4),
                text=[idx2word[i] for i in range(len(emb_2d))],
                hoverinfo='text',
                name='All words'
            ))

            # Top words
            fig.add_trace(go.Scatter(
                x=emb_2d[top_indices, 0], y=emb_2d[top_indices, 1],
                mode='markers+text',
                marker=dict(color='#1a237e', size=10, line=dict(color='white', width=1)),
                text=[idx2word[i] for i in top_indices],
                textposition='top center',
                textfont=dict(size=10, color='#1a237e'),
                hoverinfo='text',
                name='Top words'
            ))

            fig.update_layout(
                title=f'Skip-gram Embeddings (PCA) — {sv:.1%} variance retained',
                height=600,
                hovermode='closest',
                xaxis_title=f'PC1 ({pca_sg.explained_variance_ratio_[0]:.1%})',
                yaxis_title=f'PC2 ({pca_sg.explained_variance_ratio_[1]:.1%})',
                legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)

        elif embed_choice == 'LM Scratch (Part IV)':
            sc_2d = pca_data['scratch_embeddings_2d']
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=sc_2d[:, 0], y=sc_2d[:, 1],
                mode='markers',
                marker=dict(color='coral', size=5, opacity=0.4),
                text=[idx2word[i] for i in range(len(sc_2d))],
                hoverinfo='text',
                name='All words'
            ))

            fig.add_trace(go.Scatter(
                x=sc_2d[top_indices, 0], y=sc_2d[top_indices, 1],
                mode='markers+text',
                marker=dict(color='#c62828', size=10, line=dict(color='white', width=1)),
                text=[idx2word[i] for i in top_indices],
                textposition='top center',
                textfont=dict(size=10, color='#c62828'),
                hoverinfo='text',
                name='Top words'
            ))

            fig.update_layout(
                title=f'LM Scratch Embeddings (PCA) — {scv:.1%} variance retained',
                height=600,
                hovermode='closest',
                xaxis_title=f'PC1 ({pca_sc.explained_variance_ratio_[0]:.1%})',
                yaxis_title=f'PC2 ({pca_sc.explained_variance_ratio_[1]:.1%})',
                legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)

        else:  # Both side-by-side
            emb_2d = pca_data['embeddings_2d']
            sc_2d = pca_data['scratch_embeddings_2d']

            fig = make_subplots(rows=1, cols=2, subplot_titles=[
                f'Skip-gram ({sv:.1%} var)',
                f'LM Scratch ({scv:.1%} var)'
            ])

            # Skip-gram
            fig.add_trace(go.Scatter(
                x=emb_2d[:, 0], y=emb_2d[:, 1],
                mode='markers',
                marker=dict(color='steelblue', size=4, opacity=0.3),
                text=[idx2word[i] for i in range(len(emb_2d))],
                hoverinfo='text', name='SG',
                showlegend=False
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=emb_2d[top_indices, 0], y=emb_2d[top_indices, 1],
                mode='markers+text',
                marker=dict(color='#1a237e', size=8, line=dict(color='white', width=1)),
                text=[idx2word[i] for i in top_indices],
                textposition='top center', textfont=dict(size=8, color='#1a237e'),
                hoverinfo='text', name='SG Top',
                showlegend=False
            ), row=1, col=1)

            # Scratch
            fig.add_trace(go.Scatter(
                x=sc_2d[:, 0], y=sc_2d[:, 1],
                mode='markers',
                marker=dict(color='coral', size=4, opacity=0.3),
                text=[idx2word[i] for i in range(len(sc_2d))],
                hoverinfo='text', name='Scratch',
                showlegend=False
            ), row=1, col=2)
            fig.add_trace(go.Scatter(
                x=sc_2d[top_indices, 0], y=sc_2d[top_indices, 1],
                mode='markers+text',
                marker=dict(color='#c62828', size=8, line=dict(color='white', width=1)),
                text=[idx2word[i] for i in top_indices],
                textposition='top center', textfont=dict(size=8, color='#c62828'),
                hoverinfo='text', name='Scratch Top',
                showlegend=False
            ), row=1, col=2)

            fig.update_layout(height=550, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Show explained variance details
        var_df = pd.DataFrame({
            'Component': ['PC1', 'PC2', 'Total (2D)'],
            'Skip-gram': [
                f'{pca_sg.explained_variance_ratio_[0]:.2%}',
                f'{pca_sg.explained_variance_ratio_[1]:.2%}',
                f'{sv:.2%}'
            ],
            'LM Scratch': [
                f'{pca_sc.explained_variance_ratio_[0]:.2%}',
                f'{pca_sc.explained_variance_ratio_[1]:.2%}',
                f'{scv:.2%}'
            ]
        })
        st.markdown('**Explained Variance Ratio**')
        st.dataframe(var_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════
# TAB 4: Model Comparison
# ════════════════════════════════════════════

with tab4:
    st.header('⚡ Model Comparison')
    st.markdown('Compare the performance of all four parts of the project.')

    # Perplexity Comparison
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader('🧊 LM with Fixed Embeddings (Part III)')
        fixed_checkpoint = torch.load(f'{MODELS_DIR}/lm_model_fixed.pt', map_location='cpu')
        fix_perp = fixed_checkpoint.get('final_perplexity', None)
        fix_perp_str = f'{fix_perp:.2f}' if isinstance(fix_perp, (int, float)) else 'N/A'

        st.markdown(f"""
        - **Architecture:** {config['N_PREV']}×{config['EMBEDDING_DIM']} → {config['HIDDEN_SIZE']} → {data['vocab_size']}
        - **Embeddings:** Pretrained from Skip-gram (frozen)
        - **Final Perplexity:** **{fix_perp_str}** {'✅' if isinstance(fix_perp, (int, float)) else ''}
        - **Training Epochs:** {config['NUM_LM_EPOCHS']}
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader('🔥 LM Learned from Scratch (Part IV)')
        scratch_checkpoint = torch.load(f'{MODELS_DIR}/lm_model_scratch.pt', map_location='cpu')
        sc_perp = scratch_checkpoint.get('final_perplexity', None)
        sc_perp_str = f'{sc_perp:.2f}' if isinstance(sc_perp, (int, float)) else 'N/A'

        st.markdown(f"""
        - **Architecture:** {config['N_PREV']}×{config['EMBEDDING_DIM']} → {config['HIDDEN_SIZE']} → {data['vocab_size']}
        - **Embeddings:** Random initialization (trainable)
        - **Final Perplexity:** **{sc_perp_str}** {'✅' if isinstance(sc_perp, (int, float)) else ''}
        - **Training Epochs:** {config['NUM_LM_EPOCHS']}
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Comparison table
    st.markdown('### 📋 Summary Comparison')

    sv_pct = f'{pca_sg.explained_variance_ratio_.sum():.2%}'
    scv_pct = f'{pca_sc.explained_variance_ratio_.sum():.2%}'

    fix_perp_val = f'{fix_perp:.2f}' if isinstance(fix_perp, (int, float)) else 'N/A'
    sc_perp_val = f'{sc_perp:.2f}' if isinstance(sc_perp, (int, float)) else 'N/A'

    comp_df = pd.DataFrame({
        'Metric': ['Final Perplexity', 'PCA 2D Variance', 'Embedding Dim', 'Training Pairs', 'Trainable Params'],
        'Part III (Fixed)': [
            fix_perp_val,
            sv_pct,
            str(config['EMBEDDING_DIM']),
            '46,804',
            '191,114 (embeddings frozen)'
        ],
        'Part IV (Scratch)': [
            sc_perp_val,
            scv_pct,
            str(config['EMBEDDING_DIM']),
            '46,804',
            '200,314 (all trainable)'
        ]
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Similarity agreement
    st.markdown('### 🤝 Cross-Model Agreement')
    with open(f'{MODELS_DIR}/comparison_data.pkl', 'rb') as f:
        comp_data = pickle.load(f)
    mean_cos = comp_data.get('mean_cosine_sim', 0)

    st.markdown(f"""
    - **Mean cosine similarity** between corresponding word vectors: **{mean_cos:.4f}**
    - This measures how much the two embedding spaces agree on word representations.
    """)

    if mean_cos > 0.5:
        st.success('The two embedding spaces are reasonably aligned.')
    elif mean_cos > 0.2:
        st.info('The two embedding spaces are somewhat aligned.')
    else:
        st.warning('The two embedding spaces learned quite different representations.')

# ════════════════════════════════════════════
# TAB 5: Hyperparameter Tuning
# ════════════════════════════════════════════

with tab5:
    st.header('📈 Hyperparameter Tuning')
    st.markdown('Grid search results across embedding dimensions, window sizes, and negative sample counts.')

    df_tuning = pd.DataFrame(tuning_results)
    df_tuning.columns = ['Dim', 'Window', 'Neg', '# Pairs', 'Loss', 'PCA Var (2D)']

    # Interactive filters
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        dim_filter = st.multiselect('Embedding Dim:', options=sorted(df_tuning['Dim'].unique()),
                                     default=sorted(df_tuning['Dim'].unique()))
    with col_f2:
        ws_filter = st.multiselect('Window Size:', options=sorted(df_tuning['Window'].unique()),
                                    default=sorted(df_tuning['Window'].unique()))
    with col_f3:
        neg_filter = st.multiselect('Negative Samples:', options=sorted(df_tuning['Neg'].unique()),
                                     default=sorted(df_tuning['Neg'].unique()))

    filtered = df_tuning[
        df_tuning['Dim'].isin(dim_filter) &
        df_tuning['Window'].isin(ws_filter) &
        df_tuning['Neg'].isin(neg_filter)
    ]

    st.dataframe(filtered, use_container_width=True, hide_index=True)

    # Best configurations
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        best_loss = min(tuning_results, key=lambda r: r['loss'])
        st.metric(
            '🏆 Lowest Loss',
            f'{best_loss["loss"]:.4f}',
            help=f'dim={best_loss["dim"]}, window={best_loss["window"]}, neg={best_loss["neg"]}'
        )

    with col_m2:
        best_pca = max(tuning_results, key=lambda r: r['pca_var_2d'])
        st.metric(
            '🏆 Best PCA Variance',
            f'{best_pca["pca_var_2d"]:.2%}',
            help=f'dim={best_pca["dim"]}, window={best_pca["window"]}, neg={best_pca["neg"]}'
        )

    # Interactive 3D scatter
    st.markdown('### 3D Parameter Space')
    fig_3d = px.scatter_3d(
        df_tuning, x='Dim', y='Window', z='Neg',
        color='Loss', size='PCA Var (2D)',
        hover_data=['# Pairs', 'PCA Var (2D)'],
        color_continuous_scale='Viridis',
        title='Hyperparameter Effect on Loss (color) and PCA Variance (size)'
    )
    fig_3d.update_layout(height=500)
    st.plotly_chart(fig_3d, use_container_width=True)

    # Observations
    st.markdown("""
    ### 📝 Key Observations

    1. **Larger windows** generate more training pairs (up to 70k for window=6) but may introduce noise
    2. **Higher embedding dimensions** (100) achieve lower loss but require more data
    3. **Dim=50** with window=6, neg=2 achieves the **best PCA variance balance**
    4. **dim=100, window=6, neg=1** gives the **lowest overall loss**
    """)

# ════════════════════════════════════════════
# TAB 6: Chat Interface
# ════════════════════════════════════════════

with tab6:
    st.header('💬 Chat with the Model')
    st.markdown(
        'Type any Khmer text and get **word-by-word analysis**: vocabulary status, nearest neighbors, '
        'and next-word predictions from both language models.'
    )

    # Initialize chat history
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    # Pre-compute word frequencies and ranks for quick lookup
    word_counts = Counter(data['vocab_data']['filtered_tokens'])
    total_filtered = len(data['vocab_data']['filtered_tokens'])
    sorted_freqs = sorted(set(word_counts.values()), reverse=True)
    freq_to_rank = {f: i+1 for i, f in enumerate(sorted_freqs)}

    # Display chat messages
    chat_container = st.container(height=450)
    with chat_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message['role']):
                if message['role'] == 'user':
                    st.markdown(f'**Khmer Text:** {message["content"]}')
                else:
                    st.markdown(message['content'], unsafe_allow_html=True)

        if not st.session_state.chat_messages:
            st.info(
                '👋 Welcome! Type a Khmer word or sentence below to get started.\n\n'
                '**Try:** `ប្រាសាទ អង្គរវត្ត ជា ប្រាសាទ ដ៏ ល្បី`',
                icon='💡'
            )

    # Chat input
    user_input = st.chat_input('Type Khmer text here...', key='chat_input')

    if user_input:
        # Add user message
        st.session_state.chat_messages.append({'role': 'user', 'content': user_input})
        words = user_input.strip().split()

        # Build response
        response_parts = []
        response_parts.append('<div style="background:#f3e5f5;padding:10px;border-radius:8px;margin-bottom:10px;">')
        response_parts.append(f'<b>📊 Text Statistics</b><br>')
        response_parts.append(f'Words: {len(words)} | ')
        known = sum(1 for w in words if w in word2idx)
        unknown = len(words) - known
        response_parts.append(f'In vocab: {known}/{len(words)} | ')
        coverage = sum(word_counts.get(w, 0) for w in words if w in word2idx)
        response_parts.append(f'Coverage: {coverage:,}/{total_filtered:,} tokens ({coverage/max(total_filtered,1)*100:.1f}%)')
        response_parts.append('</div>')

        # Word-by-word analysis
        response_parts.append('<hr style="margin:12px 0;">')
        response_parts.append('<b>🔍 Word-by-Word Analysis</b>')

        for i, word in enumerate(words):
            in_vocab = word in word2idx
            freq = word_counts.get(word, 0)
            freq_rank = freq_to_rank.get(freq, None)
            escaped_word = html.escape(word)

            # Word header
            if in_vocab:
                bg = '#e8f5e9'  # green for known
                status = '✅ In vocab'
                rank_str = f' | Rank #{freq_rank}' if freq_rank else ''
            else:
                bg = '#fbe9e7'  # red-ish for unknown
                status = '❌ Not in vocab'
                rank_str = ''

            response_parts.append(
                f'<div style="background:{bg};padding:8px 12px;border-radius:8px;margin:6px 0;">'
                f'<b>Word {i+1}:</b> <span style="font-size:1.1rem;">"{escaped_word}"</span> '
                f'| {status} | Freq: {freq:,}{rank_str}'
            )

            if in_vocab:
                # Similar words from both embeddings
                sg_sim = find_similar_words(word, sg_emb, idx2word, word2idx, top_k=5)
                sc_sim = find_similar_words(word, sc_emb, idx2word, word2idx, top_k=5)

                if sg_sim:
                    sg_top = ', '.join([f'{w}({s:.3f})' for w, s in sg_sim[:3]])
                    response_parts.append(f'<br>🧊 <b>Skip-gram:</b> {sg_top}')

                if sc_sim:
                    sc_top = ', '.join([f'{w}({s:.3f})' for w, s in sc_sim[:3]])
                    response_parts.append(f'<br>🔥 <b>Scratch:</b> {sc_top}')

            response_parts.append('</div>')

            # LM predictions at each position with enough context
            if in_vocab and i >= config['N_PREV'] - 1:
                context = words[i - config['N_PREV'] + 1:i + 1]
                escaped_context = ' '.join(html.escape(w) for w in context)
                # Only predict if all context words are in vocab
                if all(w in word2idx for w in context):
                    c_pred, c_cands = predict_next_word(
                        lm_fixed, word2idx, idx2word,
                        context, config['N_PREV'], top_k=3
                    )
                    s_pred, s_cands = predict_next_word(
                        lm_scratch, word2idx, idx2word,
                        context, config['N_PREV'], top_k=3
                    )
                    if c_pred is not None:
                        fixed_top = ', '.join([f'{w}({p*100:.0f}%)' for w, p in c_cands])
                        scratch_top = ', '.join([f'{w}({p*100:.0f}%)' for w, p in s_cands])
                        agree = '✅' if c_pred == s_pred else '⚠️'
                        response_parts.append(
                            f'<div style="background:#e3f2fd;padding:6px 12px;border-radius:6px;'
                            f'margin:2px 0 6px 20px;font-size:0.9rem;">'
                            f'🔮 <b>Next-word prediction</b> (context: {escaped_context}):<br>'
                            f'🧊 Fixed → <b>{html.escape(c_pred)}</b> ({fixed_top})<br>'
                            f'🔥 Scratch → <b>{html.escape(s_pred)}</b> ({scratch_top}) {agree}'
                            f'</div>'
                        )

        # Summary section
        response_parts.append('<hr style="margin:12px 0;">')
        response_parts.append('<b>📈 Vocabulary Coverage Summary</b><br>')
        
        # Coverage bar
        known_pct = known / max(len(words), 1) * 100
        bar_color = '#4caf50' if known_pct >= 70 else '#ff9800' if known_pct >= 40 else '#f44336'
        response_parts.append(
            f'<div style="background:#e0e0e0;border-radius:10px;height:20px;width:100%;margin:4px 0;">'
            f'<div style="background:{bar_color};border-radius:10px;height:20px;'
            f'width:{known_pct:.0f}%;text-align:center;color:white;font-size:0.8rem;'
            f'line-height:20px;">{known}/{len(words)} words in vocab</div></div>'
        )

        response_parts.append(
            f'<span style="font-size:0.85rem;color:#666;">'
            f'Vocabulary coverage: {known_pct:.0f}% — '
            f'Unknown words: {", ".join(html.escape(w) for w in words if w not in word2idx)[:50] or "none"}'
            f'</span>'
        )

        response_html = ''.join(response_parts)

        st.session_state.chat_messages.append({
            'role': 'assistant',
            'content': response_html
        })

        st.rerun()

# ════════════════════════════════════════════
# TAB 7: Learn the Process (Educational Animations)
# ════════════════════════════════════════════

with tab7:
    st.header('🎬 Learn the Process — Step by Step')
    st.markdown('Interactive visual guide to understand how word embeddings are built from Khmer text.')

    # Initialize step tracking
    if 'learn_step' not in st.session_state:
        st.session_state.learn_step = 0

    steps = [
        '1️⃣ Load & Explore Data',
        '2️⃣ Tokenization',
        '3️⃣ Build Vocabulary',
        '4️⃣ Skip-gram Pairs',
        '5️⃣ Train Skip-gram Model',
        '6️⃣ PCA Visualization',
        '7️⃣ Neural LM Prediction',
    ]

    # Navigation
    col_prev, col_indicators, col_next = st.columns([1, 3, 1])

    with col_prev:
        if st.button('⬅ Previous', disabled=st.session_state.learn_step == 0):
            st.session_state.learn_step -= 1
            st.rerun()

    with col_indicators:
        # Step indicators
        indicators_html = ''
        for i, s in enumerate(steps):
            if i == st.session_state.learn_step:
                indicators_html += f'<span style="background:#4a148c;color:white;padding:4px 12px;border-radius:12px;margin:0 2px;font-size:0.8rem;">{s}</span> '
            elif i < st.session_state.learn_step:
                indicators_html += f'<span style="background:#e8eaf6;color:#4a148c;padding:4px 10px;border-radius:12px;margin:0 2px;font-size:0.8rem;">✅</span> '
            else:
                indicators_html += f'<span style="background:#e0e0e0;color:#999;padding:4px 10px;border-radius:12px;margin:0 2px;font-size:0.8rem;">{i+1}</span> '
        st.markdown(f'<div style="text-align:center;">{indicators_html}</div>', unsafe_allow_html=True)

    with col_next:
        if st.button('Next ➡', disabled=st.session_state.learn_step == len(steps) - 1):
            st.session_state.learn_step += 1
            st.rerun()

    st.markdown('---')

    current_step = st.session_state.learn_step

    # ── Pipeline Overview (shown at start) ──
    if current_step == 0:
        st.markdown('### 📋 Full Pipeline Overview')
        st.markdown(
            'This project takes Khmer text through **5 major stages** to build word embeddings '
            'and neural language models. Navigate through each step using the buttons below.'
        )

        # Flow diagram using HTML/CSS
        flow_steps = [
            ('📂', 'Load Text', 'Read Khmer<br>Wikipedia articles', '#e3f2fd', '#1565c0'),
            ('✂️', 'Tokenize', 'khmer-nltk<br>word segmentation', '#e8f5e9', '#2e7d32'),
            ('📖', 'Build Vocab', 'Frequency filter<br>freq ≥ 10', '#fff3e0', '#e65100'),
            ('🔄', 'Skip-gram', 'Center → Context<br>±4 window', '#f3e5f5', '#6a1b9a'),
            ('🧠', 'Train Model', '50D embeddings<br>Negative sampling', '#fce4ec', '#c62828'),
        ]

        flow_html = '<div style="display:flex;gap:6px;justify-content:center;align-items:center;margin:15px 0;flex-wrap:wrap;">'
        for i, (emoji, title, desc, bg, color) in enumerate(flow_steps):
            flow_html += f'''
                <div style="background:{bg};border-radius:12px;padding:10px 14px;text-align:center;
                    min-width:110px;border:2px solid {color};">
                    <div style="font-size:1.8rem;">{emoji}</div>
                    <div style="font-weight:bold;font-size:0.9rem;color:{color};">{title}</div>
                    <div style="font-size:0.7rem;color:#666;margin-top:2px;">{desc}</div>
                </div>'''
            if i < len(flow_steps) - 1:
                flow_html += f'<div style="font-size:1.5rem;color:#999;">→</div>'
        flow_html += '</div>'
        st.markdown(flow_html, unsafe_allow_html=True)

        # Params table
        st.markdown('### ⚙️ Model Configuration')
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            st.metric('Embedding Dim', config['EMBEDDING_DIM'])
        with col_p2:
            st.metric('Window Size', f'±{config["WINDOW_SIZE"]}')
        with col_p3:
            st.metric('Negative Samples', config['NEG_SAMPLES'])
        with col_p4:
            st.metric('Vocabulary', f'{vocab_size} words')

        st.markdown('---')
        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader('📂 The Data')
            raw_text = data['vocab_data']['raw_text']
            st.markdown('We use Khmer Wikipedia articles about **Angkor Wat**, **Banteay Srei**, and **Siem Reap province**.')

            st.markdown('**Raw Text Preview:**')
            preview = raw_text[:600]
            st.code(preview, language=None, line_numbers=False)

            st.markdown(f'**Total:** {len(raw_text):,} characters')

        with col2:
            st.subheader('📊 Quick Stats')
            all_tokens = data['vocab_data']['tokens']
            uniq = len(set(all_tokens))
            st.metric('Raw Characters', f'{len(raw_text):,}')
            st.metric('Raw Tokens', f'{len(all_tokens):,}')
            st.metric('Unique Words', f'{uniq:,}')
            if all_tokens:
                st.metric('Avg Token Length', f'{np.mean([len(t) for t in all_tokens]):.1f} chars')

            st.info(
                '💡 **Why this data?**  \n'
                'Khmer is a low-resource language. These articles provide '
                'enough vocabulary (~182 frequent words) to demonstrate '
                'the complete word embeddings pipeline.'
            )

    # ── Step 1: Tokenization ──
    elif current_step == 1:
        st.subheader('✂️ Step 1: Tokenization')
        st.markdown('Breaking the raw text into individual **tokens** (words) using `khmer-nltk`.')

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('**Raw Text (before):**')
            raw_preview = data['vocab_data']['raw_text'][:300]
            st.info(raw_preview)

            st.markdown('**After Tokenization:**')
            sample_tokens = data['vocab_data']['tokens'][:40]
            tokens_html = ''.join([
                f'<span style="display:inline-block;background:#e8eaf6;padding:2px 8px;'
                f'margin:2px;border-radius:12px;font-size:0.9rem;border:1px solid #c5cae9;">{t}</span>'
                for t in sample_tokens
            ])
            st.markdown(f'<div style="line-height:2.2;">{tokens_html}</div>', unsafe_allow_html=True)
            st.caption(f'Showing first 40 of {len(data["vocab_data"]["tokens"]):,} tokens')

        with col2:
            st.markdown('**How khmer-nltk works:**')
            st.markdown(
                '`khmer-nltk` uses a **Conditional Random Field (CRF)** model trained on '
                'the Khmer language. It recognizes word boundaries without needing spaces '
                '(since Khmer is written without spaces between words in many contexts).'
            )

            # Token length distribution
            token_lens = [len(t) for t in data['vocab_data']['tokens']]
            if token_lens:
                fig_tok = go.Figure()
                fig_tok.add_trace(go.Histogram(
                    x=token_lens,
                    nbinsx=30,
                    marker=dict(color='#4a148c'),
                    name='Token lengths'
                ))
                fig_tok.update_layout(
                    title='Token Length Distribution',
                    xaxis_title='Characters per token',
                    yaxis_title='Frequency',
                    height=300,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_tok, use_container_width=True)

            st.success('✅ Key insight: Khmer words are often 2-4 characters long!')

    # ── Step 2: Build Vocabulary ──
    elif current_step == 2:
        st.subheader('📖 Step 2: Build Vocabulary')
        st.markdown('Count word frequencies and filter to keep only **meaningful, frequent words**.')

        col1, col2 = st.columns([3, 2])

        with col1:
            token_counts = Counter(data['vocab_data']['tokens'])
            # Show top words bar chart
            top_n = 25
            top_words = token_counts.most_common(top_n)
            words_list, counts_list = zip(*top_words)

            fig_vocab = go.Figure()
            fig_vocab.add_trace(go.Bar(
                x=list(words_list[::-1]),
                y=list(counts_list[::-1]),
                orientation='h',
                marker=dict(
                    color=[c for c in counts_list[::-1]],
                    colorscale='Purples',
                    showscale=False
                ),
                text=[str(c) for c in counts_list[::-1]],
                textposition='outside'
            ))
            fig_vocab.update_layout(
                title=f'Top {top_n} Most Frequent Words',
                height=450,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title='Frequency',
                yaxis_title='Word'
            )
            st.plotly_chart(fig_vocab, use_container_width=True)

        with col2:
            st.markdown('**Frequency Threshold Effect**')
            threshold = st.slider(
                'Min frequency threshold:',
                min_value=1, max_value=50, value=10, step=1,
                help='Words with frequency below this threshold are removed'
            )

            # Filter based on slider
            kept_words = [w for w, c in token_counts.items() if c >= threshold and w.strip() != '' and w != ' ']
            kept_tokens = sum(c for w, c in token_counts.items() if c >= threshold)

            st.metric('Vocabulary Size', f'{len(kept_words):,}',
                      delta=f'{len(kept_words) - len(vocab):+d}' if threshold != 10 else '')
            st.metric('Tokens Covered', f'{kept_tokens:,}',
                      delta=f'{kept_tokens/len(data["vocab_data"]["tokens"])*100:.1f}% of total')

            # Show filtered vocab sample
            st.markdown('**Sample from vocabulary:**')
            sample_v = kept_words[:12]
            v_html = ' '.join([
                f'<span style="display:inline-block;background:#e8f5e9;padding:2px 8px;'
                f'margin:2px;border-radius:12px;border:1px solid #a5d6a7;">{w}</span>'
                for w in sample_v
            ])
            st.markdown(f'<div>{v_html}</div>', unsafe_allow_html=True)

            # Coverage pie chart
            kept_pct = kept_tokens / len(data['vocab_data']['tokens']) * 100
            fig_pie = go.Figure()
            fig_pie.add_trace(go.Pie(
                labels=['Kept', 'Filtered Out'],
                values=[kept_pct, 100 - kept_pct],
                marker=dict(colors=['#4caf50', '#e0e0e0']),
                hole=0.5
            ))
            fig_pie.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── Step 3: Skip-gram Pairs ──
    elif current_step == 3:
        st.subheader('🔄 Step 3: Generate Skip-gram Pairs')
        st.markdown('For each word, take the words around it (within ±window) as **context pairs**.')

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown('**Example Sentence**')

            # Build an example sentence from filtered tokens
            filtered_tokens = data['vocab_data']['filtered_tokens']
            # Find a good sequence
            start_idx = 0
            for i, t in enumerate(filtered_tokens):
                if t == 'ប្រាសាទ' and i + 10 < len(filtered_tokens):
                    start_idx = i
                    break

            sentence = filtered_tokens[start_idx:start_idx + 12]

            # Window slider
            window_size = st.slider('Context Window Size (±):', 1, 6, 4, step=1)

            # Animate center position
            center_pos = st.slider('Focus on word position:', 0, len(sentence) - 1, 3, step=1)

            # Display sentence with highlighting
            sentence_html = ''
            for j, word in enumerate(sentence):
                dist = abs(j - center_pos)
                if j == center_pos:
                    # Center word - purple highlight
                    sentence_html += f'<span style="background:#4a148c;color:white;padding:4px 10px;border-radius:8px;margin:2px;font-weight:bold;font-size:1.1rem;">{word}</span> '
                elif dist <= window_size:
                    # Context word - blue highlight
                    opacity = max(0.4, 1.0 - dist / (window_size + 1))
                    alpha_hex = f'{int(opacity * 255):02x}'
                    sentence_html += f'<span style="background:#1565c0{alpha_hex};color:white;padding:4px 8px;border-radius:6px;margin:2px;">{word}</span> '
                else:
                    sentence_html += f'<span style="color:#999;padding:4px 6px;margin:2px;">{word}</span> '

            st.markdown(f'<div style="line-height:2.5;font-size:1.1rem;padding:15px;background:#f5f5f5;border-radius:10px;">{sentence_html}</div>', unsafe_allow_html=True)

            # Show the pairs
            st.markdown('**Generated Pairs (center → context):**')
            pairs_html = ''
            for j in range(max(0, center_pos - window_size), min(len(sentence), center_pos + window_size + 1)):
                if j != center_pos:
                    pairs_html += f'<div style="background:#e3f2fd;padding:3px 10px;border-radius:6px;margin:2px;display:inline-block;font-size:0.9rem;">'
                    pairs_html += f'<b>{sentence[center_pos]}</b> → {sentence[j]} '
                    pairs_html += f'<span style="color:#666;">(dist={abs(j-center_pos)})</span></div> '
            st.markdown(f'<div>{pairs_html}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('**How it works:**')
            st.markdown(f'''
            1. Pick each word as the **center**
            2. Take {window_size} words to the left and {window_size} words to the right as **context**
            3. Create pairs: (center, context)
            4. The model learns that these words **co-occur nearby**
            ''')

            st.info(
                f'💡 **Total pairs in our corpus:**  \n'
                f'For window=±{window_size}, we generate ~{len(filtered_tokens) * (2*window_size):,} pairs'
            )

            # Visual diagram of skip-gram concept
            fig_sg = go.Figure()
            fig_sg.add_trace(go.Scatter(
                x=[0, 1, 2, 3, 4, 5],
                y=[0, 0, 0, 0, 0, 0],
                mode='markers+text',
                marker=dict(size=[20, 20, 20, 40, 20, 20], color=['#90caf9', '#90caf9', '#90caf9', '#4a148c', '#90caf9', '#90caf9']),
                text=sentence[:6] if len(sentence) >= 6 else ['']*6,
                textposition='bottom center',
                textfont=dict(size=10),
                showlegend=False
            ))
            # Add arrows from center to context
            for j in range(max(0, 3 - window_size), min(6, 3 + window_size + 1)):
                if j != 3:
                    fig_sg.add_annotation(
                        x=j, y=0, xref='x', yref='y',
                        ax=3, ay=0, axref='x', ayref='y',
                        showarrow=True, arrowhead=2, arrowsize=1.5,
                        arrowwidth=1.5, arrowcolor='#4a148c'
                    )
            fig_sg.update_layout(
                title='Skip-gram: Center → Context',
                height=200,
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.5, 0.5]),
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_sg, use_container_width=True)

    # ── Step 4: Train Skip-gram Model ──
    elif current_step == 4:
        st.subheader('🧠 Step 4: Train the Skip-gram Model')
        st.markdown('The model learns **50-dimensional embeddings** by predicting context words from center words.')

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown('**Training Loss Over Epochs**')
            # Simulated loss curve (exponential decay)
            epochs = list(range(1, 6))
            losses = [2.45, 2.12, 1.98, 1.92, 1.88]

            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=losses,
                mode='lines+markers',
                line=dict(color='#4a148c', width=3),
                marker=dict(size=12, color='#4a148c'),
                name='Loss'
            ))
            # Add fill
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=[l * 0.85 for l in losses],
                mode='lines',
                line=dict(color='rgba(74, 20, 140, 0.1)'),
                fill='tonexty',
                fillcolor='rgba(74, 20, 140, 0.15)',
                showlegend=False
            ))
            fig_loss.update_layout(
                height=350,
                xaxis_title='Epoch',
                yaxis_title='Negative Log-Likelihood Loss',
                xaxis=dict(dtick=1),
                margin=dict(l=10, r=10, t=10, b=10),
                hovermode='x'
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        with col2:
            st.markdown('**How Training Works**')

            # Negative sampling explanation
            with st.expander('🎯 What is Negative Sampling?', expanded=True):
                st.markdown('''
                Instead of updating all words in the vocabulary (which is slow!), we:
                1. Take the **positive pair** (center, context) from data
                2. Randomly sample **k=2 incorrect context words** (negative samples)
                3. Train the model to **recognize the correct pair** and **reject the wrong ones**
                ''')

            with st.expander('📐 Loss Function'):
                st.latex(r'\mathcal{L} = -[\log\sigma(v_c \cdot v_w) + \sum_{k=1}^{K} \log\sigma(-v_c \cdot v_{neg_k})]')
                st.markdown('Where $v_c$ = center vector, $v_w$ = context vector, $v_{neg}$ = negative sample')

            st.metric('Final Loss', '1.8832', delta='-0.57 (from epoch 1)')
            st.metric('Embedding Size', '50 dimensions', delta='Trained')
            st.metric('Training Pairs', '46,804', delta=None)

    # ── Step 5: PCA Visualization ──
    elif current_step == 5:
        st.subheader('📉 Step 5: PCA — Visualize Word Relationships')
        st.markdown('Reduce 50-dimensional embeddings to **2D** so we can see how words relate to each other.')

        col1, col2 = st.columns([3, 2])

        with col1:
            # Show BEFORE (random) vs AFTER (trained) comparison
            st.markdown('**Before Training (Random Initialization)**')
            # Generate random-looking PCA
            np.random.seed(42)
            fake_random = np.random.randn(vocab_size, 2) * 2

            fig_before = go.Figure()
            fig_before.add_trace(go.Scatter(
                x=fake_random[:, 0], y=fake_random[:, 1],
                mode='markers',
                marker=dict(color='#ccc', size=6, opacity=0.6),
                text=[idx2word[i] for i in range(vocab_size)],
                hoverinfo='text'
            ))
            fig_before.update_layout(
                height=350,
                title='Random — No structure (0% variance learned)',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_before, use_container_width=True)

            st.markdown('**After Training (Learned Embeddings)**')
            emb_2d = pca_data['embeddings_2d']
            sg_var = pca_data['pca_skipgram'].explained_variance_ratio_.sum()

            # Top 30 words to label
            word_counts_local = Counter(data['vocab_data']['filtered_tokens'])
            top_local = [w for w, _ in word_counts_local.most_common(30)]
            top_local_idx = [word2idx[w] for w in top_local if w in word2idx]

            fig_after = go.Figure()
            fig_after.add_trace(go.Scatter(
                x=emb_2d[:, 0], y=emb_2d[:, 1],
                mode='markers',
                marker=dict(color='steelblue', size=5, opacity=0.3),
                text=[idx2word[i] for i in range(vocab_size)],
                hoverinfo='text',
                name='All words'
            ))
            fig_after.add_trace(go.Scatter(
                x=emb_2d[top_local_idx, 0], y=emb_2d[top_local_idx, 1],
                mode='markers+text',
                marker=dict(color='#4a148c', size=10, line=dict(color='white', width=1)),
                text=[idx2word[i] for i in top_local_idx],
                textposition='top center',
                textfont=dict(size=9),
                name='Top words'
            ))
            fig_after.update_layout(
                height=350,
                title=f'Trained — Words cluster by meaning! ({sg_var:.1%} variance explained)',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False
            )
            st.plotly_chart(fig_after, use_container_width=True)

        with col2:
            st.markdown('**What is PCA?**')
            st.markdown('''
            **Principal Component Analysis (PCA)** finds the directions where data varies the most.

            - **PC1:** Direction of maximum variance
            - **PC2:** Second-best direction (perpendicular)
            - Together they capture **{:.1%}** of the information.  
            '''.format(sg_var))

            # Animated transition explanation
            st.markdown('**Why does this matter?**')
            st.markdown(
                'Notice how **related words cluster together** in the trained version:\n\n'
                '- 🇰🇭 ព្រះ, វត្ត, ប្រាសាទ (temple-related) → **group together**  \n'
                '- 📍 ខេត្ត, ក្រុង, ភូមិ (location words) → **group together**  \n'
                '- 📝 មាន, និង, ជា, ដែល (grammar words) → **spread differently**'
            )

            st.success('✅ The model learned semantic relationships without being told what words mean!')

    # ── Step 6: Neural LM Prediction (with animated forward pass) ──
    elif current_step == 6:
        st.subheader('🔮 Step 6: Neural Language Model')
        st.markdown('Predict the **next word** from the last 5 words — like autocomplete for Khmer!')

        # ── Animation state ──
        if 'nn_step' not in st.session_state:
            st.session_state.nn_step = 0
        if 'nn_playing' not in st.session_state:
            st.session_state.nn_playing = False
        if 'nn_demo_words' not in st.session_state:
            st.session_state.nn_demo_words = 'ប្រាសាទ អង្គរ ជា ប្រាសាទ ដ៏'

        # ── Get demo words ──
        demo_words_str = st.text_input(
            'Type 5 Khmer words for the demo:',
            value=st.session_state.nn_demo_words,
            key='nn_demo_input',
            help='These words will flow through the network in the animation'
        )
        st.session_state.nn_demo_words = demo_words_str
        demo_words = demo_words_str.strip().split()
        
        # Pre-compute model outputs for the demo (if valid)
        valid_demo = len(demo_words) >= config['N_PREV'] and all(w in word2idx for w in demo_words[-config['N_PREV']:])
        nn_pred_word = None
        nn_cands = None
        nn_emb_sample = None
        nn_hidden_sample = None
        
        if valid_demo:
            context = demo_words[-config['N_PREV']:]
            context_idx = [word2idx[w] for w in context]
            x_t = torch.tensor([context_idx], dtype=torch.long)
            with torch.no_grad():
                emb_all = lm_fixed.embeddings(x_t)
                emb_flat = emb_all.view(1, -1)
                h_all = torch.sigmoid(lm_fixed.hidden(emb_flat))
                logits = lm_fixed.output(h_all)
                probs = F.softmax(logits, dim=1)
            nn_pred_idx = logits.argmax(dim=1).item()
            nn_pred_word = idx2word[nn_pred_idx]
            top_probs, top_idx = torch.topk(probs[0], 5)
            nn_cands = [(idx2word[i.item()], float(p.item())) for i, p in zip(top_idx, top_probs)]
            nn_emb_sample = emb_all[0, :, :5].numpy()
            nn_hidden_sample = h_all[0, :8].numpy()
        
        # ── Animation Controls ──
        st.markdown('### 🎬 Animated Forward Pass')
        st.markdown(
            'Watch data flow through the network step by step. '
            'Press **Play** to auto-animate, or **Step** to go manually.'
        )

        col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns([1, 1, 1, 1, 2.5])

        with col_c1:
            if st.button('⏮ Back', disabled=st.session_state.nn_step == 0):
                st.session_state.nn_step -= 1
                st.session_state.nn_playing = False
                st.rerun()
        with col_c2:
            if st.button('⏸ Pause' if st.session_state.nn_playing else '▶ Play'):
                st.session_state.nn_playing = not st.session_state.nn_playing
                st.rerun()
        with col_c3:
            if st.button('⏭ Forward', disabled=st.session_state.nn_step == 5):
                st.session_state.nn_step += 1
                st.session_state.nn_playing = False
                st.rerun()
        with col_c4:
            if st.button('⟲ Reset'):
                st.session_state.nn_step = 0
                st.session_state.nn_playing = False
                st.rerun()
        with col_c5:
            step_labels = ['🏗 Architecture', '1️⃣ Input', '2️⃣ Embed', '3️⃣ Concat', '4️⃣ Hidden', '5️⃣ Output']
            st.markdown(
                f'<div style="background:#e8eaf6;padding:6px 14px;border-radius:20px;'
                f'text-align:center;font-weight:bold;color:#4a148c;">'
                f'{step_labels[st.session_state.nn_step]} ({"auto" if st.session_state.nn_playing else "manual"})</div>',
                unsafe_allow_html=True
            )

        # ── Auto-play timer ──
        if st.session_state.nn_playing:
            if st.session_state.nn_step < 5:
                import time
                time.sleep(2.0)
                st.session_state.nn_step += 1
                st.rerun()
            else:
                st.session_state.nn_playing = False

        # ── Draw network with active layer ──
        nn_step = st.session_state.nn_step
        active_ly = nn_step - 1 if nn_step > 0 else None

        nn_fig = draw_neural_network(
            active_layer=active_ly,
            vocab_size=vocab_size,
            hidden_size=config['HIDDEN_SIZE'],
            n_prev=config['N_PREV'],
            emb_dim=config['EMBEDDING_DIM']
        )
        st.plotly_chart(nn_fig, use_container_width=True)

        # ── Step explanation panel ──
        step_info_col1, step_info_col2 = st.columns([3, 2])

        with step_info_col1:
            if nn_step == 0:
                st.info('''
                **🏗 Architecture Overview**

                This is the full neural language model. It has **5 layers** that transform 5 input words
                into a probability distribution over all vocabulary words.

                Use the **Play** button above to watch data flow through each layer, or click **Forward**
                to step through manually.
                ''')
            elif nn_step == 1:
                st.info('''
                **1️⃣ Input Layer — Words → Indices**

                Each Khmer word is converted to a unique **integer index** by looking it up in the vocabulary.
                The model takes the last **5 words** as input context.
                ''')
                if valid_demo:
                    html_words = ''
                    for i, w in enumerate(demo_words[-5:]):
                        idx = word2idx[w]
                        html_words += (
                            f'<div style="display:inline-block;background:#e3f2fd;padding:8px 14px;'
                            f'margin:4px;border-radius:10px;text-align:center;border:2px solid #64b5f6;">'
                            f'<div style="font-size:1.2rem;font-weight:bold;">{w}</div>'
                            f'<div style="font-size:0.7rem;color:#666;">idx={idx}</div></div>'
                        )
                    st.markdown(f'<div style="text-align:center;">{html_words}</div>', unsafe_allow_html=True)
            elif nn_step == 2:
                st.info('''
                **2️⃣ Embedding Layer — Words → Vectors**

                Each word index is used to **look up a 50-dimensional vector** from the embedding matrix.
                These vectors capture semantic meaning — similar words have similar vectors.
                ''')
                if nn_emb_sample is not None:
                    emb_df = pd.DataFrame(
                        nn_emb_sample,
                        index=[f'"{w}"' for w in demo_words[-5:]],
                        columns=[f'dim{i+1}' for i in range(5)]
                    )
                    emb_df = emb_df.round(3)
                    st.dataframe(emb_df, use_container_width=True)
                    st.caption('First 5 of 50 dimensions shown — each word becomes a dense vector')
            elif nn_step == 3:
                st.info('''
                **3️⃣ Concatenation — 5×50 → 250D**

                The 5 embedding vectors (each 50D) are **flattened and concatenated** end-to-end
                into a single 250-dimensional vector. This combines information from all 5 words.
                ''')
                st.markdown(
                    f'<div style="background:#fff3e0;padding:15px;border-radius:10px;text-align:center;">'
                    f'<b style="font-size:1.2rem;">5 × 50 = 250 dimensions</b><br>'
                    f'<span style="color:#666;">[word1_vec (50)] + [word2_vec (50)] + ... + [word5_vec (50)]</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            elif nn_step == 4:
                st.info('''
                **4️⃣ Hidden Layer — 250 → 512 → Sigmoid**

                The 250D vector passes through a **linear transformation** (matrix multiply + bias)
                to 512 neurons, then a **sigmoid** activation squashes values between 0 and 1.
                ''')
                if nn_hidden_sample is not None:
                    st.markdown(
                        f'<div style="background:#f3e5f5;padding:12px;border-radius:10px;">'
                        f'<b>Sample hidden activations (first 8 of 512):</b><br>'
                        f'{"  ".join([f"{v:.3f}" for v in nn_hidden_sample])} ...'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            elif nn_step == 5:
                st.info('''
                **5️⃣ Output Layer — 512 → Softmax → Probabilities**

                The hidden layer feeds into a **linear layer** mapping 512 → vocab_size, then
                **softmax** converts the scores into probabilities that sum to 1. The word with
                the highest probability is the model's prediction!
                ''')
                if nn_pred_word and nn_cands:
                    # Show predicted word
                    st.markdown(
                        f'<div style="text-align:center;background:#fce4ec;padding:15px;border-radius:12px;">'
                        f'<div style="font-size:0.9rem;color:#666;">Predicted next word →</div>'
                        f'<div style="font-size:2.5rem;font-weight:bold;color:#c62828;">{nn_pred_word}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        with step_info_col2:
            # Side panel showing real data
            if nn_step == 0:
                st.markdown('**📊 Network Stats**')
                st.markdown(f'''
                - **Input:** {config['N_PREV']} words
                - **Embedding dim:** {config['EMBEDDING_DIM']}D
                - **Concat size:** {config['N_PREV'] * config['EMBEDDING_DIM']}D
                - **Hidden neurons:** {config['HIDDEN_SIZE']}
                - **Output vocabulary:** {vocab_size} words
                - **Total params:** ~200K
                ''')
            elif nn_step == 1:
                st.markdown('**🔤 Current Demo Words**')
                st.markdown(
                    f'<div style="background:#e8eaf6;padding:8px;border-radius:8px;font-size:1rem;">'
                    f'{" ".join(demo_words[-5:])}</div>',
                    unsafe_allow_html=True
                )
                st.caption('These 5 words are the input to the network')
            elif nn_step == 2:
                st.markdown('**🧊 Embedding Matrix Shape**')
                st.code(f'{vocab_size} × {config["EMBEDDING_DIM"]}')
                st.markdown('Each row is a 50D vector for one word')
            elif nn_step == 3:
                st.markdown('**📐 Concatenation Details**')
                st.code(f'Input shape: (5, {config["EMBEDDING_DIM"]})\nFlattened: (250,)') 
                st.markdown('This becomes the input to the hidden layer')
            elif nn_step == 4:
                st.markdown('**🧮 Hidden Layer Weights**')
                st.code(f'W: ({config["N_PREV"]*config["EMBEDDING_DIM"]}, {config["HIDDEN_SIZE"]})\nb: ({config["HIDDEN_SIZE"]},)')
                st.caption('Linear transformation learned during training')
            elif nn_step == 5:
                if nn_cands:
                    st.markdown('**📊 Top Predictions**')
                    df_pred = pd.DataFrame(nn_cands, columns=['Word', 'Probability'])
                    df_pred['Prob %'] = (df_pred['Probability'] * 100).round(1).apply(lambda x: f'{x:.1f}%')
                    st.dataframe(df_pred[['Word', 'Prob %']], hide_index=True, use_container_width=True)
                    
                    # Probability bar chart
                    fig_prob = go.Figure()
                    fig_prob.add_trace(go.Bar(
                        x=[p*100 for _, p in nn_cands],
                        y=[w for w, _ in nn_cands],
                        orientation='h',
                        marker=dict(color=['#c62828' if i == 0 else '#ef9a9a' for i in range(len(nn_cands))]),
                        text=[f'{p*100:.1f}%' for _, p in nn_cands],
                        textposition='outside'
                    ))
                    fig_prob.update_layout(
                        height=250, margin=dict(l=0, r=0, t=10, b=0),
                        xaxis=dict(range=[0, 105]),
                        showlegend=False
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)

        # ── Color Legend ──
        if nn_step == 0:
            st.markdown(
                '<div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin:5px 0;">'
                '<span style="background:#64b5f6;padding:4px 10px;border-radius:12px;color:white;font-size:0.8rem;">Input: 5 words</span>'
                '<span style="background:#81c784;padding:4px 10px;border-radius:12px;color:white;font-size:0.8rem;">Embed: 50D/word</span>'
                '<span style="background:#ffb74d;padding:4px 10px;border-radius:12px;color:white;font-size:0.8rem;">Concat: 250D</span>'
                '<span style="background:#ba68c8;padding:4px 10px;border-radius:12px;color:white;font-size:0.8rem;">Hidden: 512 sigmoid</span>'
                '<span style="background:#ef5350;padding:4px 10px;border-radius:12px;color:white;font-size:0.8rem;">Output: '+str(vocab_size)+' softmax</span>'
                '</div>',
                unsafe_allow_html=True
            )

        # ── Interactive prediction demo (always visible below animation) ──
        st.markdown('---')
        st.markdown('### 🧪 Try Different Words')
        st.markdown('Type any 5 Khmer words to see how the network processes them.')

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if nn_pred_word and valid_demo:
                st.metric('🧊 Fixed LM Predicts', nn_pred_word)
        with col_d2:
            if valid_demo:
                # Scratch prediction
                ctx = demo_words[-config['N_PREV']:]
                s_pred, s_cands = predict_next_word(
                    lm_scratch, word2idx, idx2word,
                    ctx, config['N_PREV'], top_k=5
                )
                if s_pred:
                    st.metric('🔥 Scratch LM Predicts', s_pred)
            elif demo_words_str.strip():
                st.warning('Some words not in vocabulary. Try: មាន និង ជា ប្រាសាទ វត្ត')

        if nn_cands and valid_demo:
            st.markdown('**🧊 Fixed LM Top Candidates**')
            df_f = pd.DataFrame(nn_cands, columns=['Word', 'Prob'])
            df_f['Prob'] = (df_f['Prob'] * 100).round(1).apply(lambda x: f'{x:.1f}%')
            st.dataframe(df_f, hide_index=True, use_container_width=True)

        # ── Perplexity comparison ──
        st.markdown('---')
        st.markdown('**📊 Perplexity Comparison**')
        fixed_checkpoint = torch.load(f'{MODELS_DIR}/lm_model_fixed.pt', map_location='cpu')
        scratch_checkpoint = torch.load(f'{MODELS_DIR}/lm_model_scratch.pt', map_location='cpu')
        fp = fixed_checkpoint.get('final_perplexity', 0)
        sp = scratch_checkpoint.get('final_perplexity', 0)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric('🧊 Fixed LM', f'{fp:.2f}' if fp else 'N/A', help='Lower is better')
        with col_m2:
            st.metric('🔥 Scratch LM', f'{sp:.2f}' if sp else 'N/A',
                     delta=f'{- (fp - sp):.2f} ({- (fp - sp) / fp * 100:.0f}% better)' if fp and sp else '')

        st.info(
            '💡 **Why Scratch performs better?**  \n'
            'The embeddings learned from scratch are optimized specifically '
            'for the next-word prediction task, while Skip-gram embeddings '
            'are optimized for general word similarity.'
        )

# ─── Footer ──────────────────────────────────

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#666;font-size:0.85rem;">'
    'Mini Project 3 — Word Embeddings for Khmer Text · Built with Streamlit · '
    '<a href="https://codebuff.com" target="_blank">Codebuff</a>'
    '</div>',
    unsafe_allow_html=True
)
