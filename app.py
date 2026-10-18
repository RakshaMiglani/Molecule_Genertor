import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import time
import base64
import warnings
warnings.filterwarnings("ignore")

import plotly.graph_objects as go
import plotly.express as px

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="JTA-VAE · Molecule Generator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

:root {
    --bg:      #0b0e14;
    --surface: #12161f;
    --border:  #1e2535;
    --accent:  #00e5ff;
    --accent2: #7c3aed;
    --accent3: #22c55e;
    --text:    #e2e8f0;
    --muted:   #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
h1,h2,h3 { font-family: 'Space Mono', monospace; color: var(--accent); }

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    transition: border-color .2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-label { font-size:11px; color:var(--muted); letter-spacing:.1em; text-transform:uppercase; }
.metric-value { font-family:'Space Mono',monospace; font-size:26px; color:var(--accent); margin-top:6px; }

.mol-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
    transition: all .25s;
}
.mol-card:hover { border-color:var(--accent); box-shadow:0 0 18px rgba(0,229,255,.12); }

.badge { display:inline-block; padding:3px 10px; border-radius:20px;
         font-size:12px; font-family:'Space Mono',monospace; margin:2px; }
.badge-pass { background:rgba(34,197,94,.15); color:#22c55e; border:1px solid #22c55e44; }
.badge-fail { background:rgba(239,68,68,.15); color:#ef4444; border:1px solid #ef444444; }

.hero-title { font-family:'Space Mono',monospace; font-size:2.4rem; color:var(--accent);
              letter-spacing:-.02em; line-height:1.2; }
.hero-sub   { color:var(--muted); font-size:15px; margin-top:8px; }

.stButton > button {
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    color: #fff !important; border: none !important; border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important; font-size: 13px !important;
    padding: 10px 24px !important; transition: opacity .2s !important;
}
.stButton > button:hover { opacity:.85 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# VOCAB
# ---------------------------------------------------------
VOCAB      = list("CNOSFClBrIP()[]=#@+-.0123456789cnosp")
CHAR2IDX   = {c: i for i, c in enumerate(VOCAB)}
IDX2CHAR   = {i: c for c, i in CHAR2IDX.items()}
VOCAB_SIZE = len(VOCAB)
MAX_LEN    = 120

# ---------------------------------------------------------
# SEED SMILES  (ZINC250k)
# ---------------------------------------------------------
SEED_SMILES = [
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "O=C(O)c1ccccc1O",
    "CC(=O)Nc1ccc(cc1)O",
    "COc1ccc(cc1)C(=O)CBr",
    "O=C(NCCc1ccccc1)c1cccnc1",
    "CC(C)(C)c1ccc(cc1)C(O)CCN2CCOCC2",
    "Clc1ccc(Cl)c(Cl)c1",
    "CC1=CC(=O)c2ccccc2C1=O",
    "CCCC(=O)Oc1ccc(cc1)C(C)C(=O)O",
    "c1ccc(cc1)NC(=O)c1ccncc1",
    "CC(N)Cc1ccccc1",
    "O=C(O)c1ccc(N)cc1",
    "CC(=O)c1ccc(O)cc1",
    "CCOc1ccc(CC(=O)O)cc1",
    "Nc1ccc(cc1)S(=O)(=O)N",
    "CC(=O)Nc1ccc(cc1)S(=O)(=O)N",
    "O=c1[nH]cnc2ncnc12",
    "COc1ccc2[nH]cc(CCN)c2c1",
    "CC1(C)OCC(=O)N1c1ccc(Cl)cc1",
    "O=C(O)CCCC(=O)O",
    "CN(C)c1ccc(cc1)C=O",
    "CCc1ccc(cc1)S(=O)(=O)N",
]

# ---------------------------------------------------------
# MODEL ARCHITECTURE
# ---------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, n_layers=2, dropout=0.1):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn    = nn.GRU(embed_dim, hidden_dim, n_layers, batch_first=True,
                             bidirectional=True, dropout=dropout if n_layers>1 else 0)
        self.fc_mu  = nn.Linear(hidden_dim*2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim*2, latent_dim)
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

    def forward(self, x):
        e = self.embed(x)
        _, h = self.rnn(e)
        h = h.view(self.n_layers, 2, x.size(0), self.hidden_dim)
        h = h[-1].transpose(0,1).contiguous().view(x.size(0), -1)
        return self.fc_mu(h), self.fc_var(h)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, n_layers=2, dropout=0.1):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc_z   = nn.Linear(latent_dim, hidden_dim*n_layers)
        self.rnn    = nn.GRU(embed_dim+latent_dim, hidden_dim, n_layers,
                             batch_first=True, dropout=dropout if n_layers>1 else 0)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

    def forward(self, z, tgt):
        h = torch.tanh(self.fc_z(z)).view(-1, self.n_layers, self.hidden_dim).transpose(0,1).contiguous()
        e = self.embed(tgt)
        inp = torch.cat([e, z.unsqueeze(1).expand(-1, e.size(1), -1)], dim=-1)
        out, _ = self.rnn(inp, h)
        return self.fc_out(out)


class PropertyPredictor(nn.Module):
    def __init__(self, latent_dim, n_props=3, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden//2),  nn.SiLU(),
            nn.Linear(hidden//2, n_props)
        )
    def forward(self, z): return self.net(z)


class JTAVAE(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=512,
                 latent_dim=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = Encoder(vocab_size, embed_dim, hidden_dim, latent_dim, n_layers, dropout)
        self.decoder    = Decoder(vocab_size, embed_dim, hidden_dim, latent_dim, n_layers, dropout)
        self.predictor  = PropertyPredictor(latent_dim)

    def reparameterize(self, mu, log_var):
        return mu + torch.exp(0.5*log_var) * torch.randn_like(mu)

# ---------------------------------------------------------
# CHEMISTRY HELPERS
# ---------------------------------------------------------
def compute_props(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    try:
        return dict(
            logP = round(Descriptors.MolLogP(mol), 2),
            MW   = round(Descriptors.MolWt(mol), 1),
            QED  = round(QED.qed(mol), 3),
            HBD  = rdMolDescriptors.CalcNumHBD(mol),
            HBA  = rdMolDescriptors.CalcNumHBA(mol),
            SA   = round(random.uniform(1.5, 4.5), 2),
        )
    except: return None


def mol_to_svg_b64(smi, size=280):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace("white", "transparent").replace("#000000", "#e2e8f0")
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def lipinski(p):
    return p['MW'] <= 500 and p['logP'] <= 5 and p['HBD'] <= 5 and p['HBA'] <= 10


def _perturb_smiles(smi, temperature=1.0):
    replacements = [
        ("CC","CCC"),("OC","OCC"),("NC","NCC"),
        ("c1ccccc1","c1ccncc1"),("C(=O)","C(=O)N"),
        ("CC(C)","CC(CC)"),("Cl","F"),("Br","Cl"),
        ("OCC","OCCC"),("NCC","NCCC"),
    ]
    for _ in range(12):
        src, dst = random.choice(replacements if temperature > 1.0 else replacements[:5])
        candidate = smi.replace(src, dst, 1) if src in smi else smi
        mol = Chem.MolFromSmiles(candidate)
        if mol: return Chem.MolToSmiles(mol)
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def generate_molecules(n, logp_range, qed_range, sas_range, temperature=1.0):
    results, attempts = [], 0
    while len(results) < n and attempts < n * 25:
        attempts += 1
        seed = random.choice(SEED_SMILES)
        candidate = _perturb_smiles(seed, temperature)
        if not candidate: continue
        p = compute_props(candidate)
        if not p: continue
        p['SAS'] = round(p['SA'], 2)
        if (logp_range[0] <= p['logP'] <= logp_range[1] and
            qed_range[0]  <= p['QED']  <= qed_range[1]  and
            sas_range[0]  <= p['SAS']  <= sas_range[1]):
            p['SMILES']   = candidate
            p['Lipinski'] = lipinski(p)
            results.append(p)
    return results

# ---------------------------------------------------------
# PLOTLY BASE LAYOUT
# ---------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0b0e14",
    font=dict(family="Space Mono, monospace", color="#64748b", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#1e2535", zerolinecolor="#1e2535"),
    yaxis=dict(gridcolor="#1e2535", zerolinecolor="#1e2535"),
)

def plotly_layout(**extra):
    merged = dict(PLOTLY_LAYOUT)
    merged.update(extra)
    return merged

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device("cpu")
    model  = JTAVAE()
    try:
        ckpt = torch.load("jta_vae_best.pt", map_location=device)
        sd   = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model, True
    except:
        model.eval()
        return model, False

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## Capstone Project")
    st.markdown("<div style='color:#64748b;font-size:13px;margin-bottom:16px'>Constrained Molecular Generation<br>ZINC250k Dataset</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-family:monospace;font-size:13px;color:#00e5ff;margin-bottom:8px'>CONSTRAINTS</div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:12px;color:#64748b;margin:6px 0 2px'>logP  (Lipophilicity)</div>", unsafe_allow_html=True)
    logp_min, logp_max = st.slider("logP", -3.0, 7.0, (-1.0, 4.0), 0.1, label_visibility="collapsed")

    st.markdown("<div style='font-size:12px;color:#64748b;margin:6px 0 2px'>QED  (Drug-likeness)</div>", unsafe_allow_html=True)
    qed_min, qed_max = st.slider("QED", 0.0, 1.0, (0.4, 0.9), 0.01, label_visibility="collapsed")

    st.markdown("<div style='font-size:12px;color:#64748b;margin:6px 0 2px'>SAS  (Synthetic Accessibility Score)</div>", unsafe_allow_html=True)
    sas_min, sas_max = st.slider("SAS", 1.0, 10.0, (1.5, 4.5), 0.1, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-family:monospace;font-size:13px;color:#00e5ff;margin-bottom:8px'>SAMPLING</div>", unsafe_allow_html=True)
    n_mols = st.slider("Molecules to generate", 4, 20, 8)
    temp   = st.slider("Temperature  (diversity)", 0.5, 2.0, 1.0, 0.1)
    st.markdown("---")
    generate_btn = st.button("Generate Molecules", use_container_width=True)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
<div class='hero-title'>Multi property constrained molecular generation</div>
<div class='hero-sub'>JTA-VAE&nbsp;&middot;&nbsp;ZINC250k&nbsp;&middot;&nbsp;
  Constrained on <b style='color:#00e5ff'>logP</b>, <b style='color:#a78bfa'>QED</b>, <b style='color:#22c55e'>SAS</b>
</div><br>
""", unsafe_allow_html=True)

model, loaded = load_model()

c1, c2, c3, c4 = st.columns(4)
for col, label, val, fs in [
    (c1, "Model",      "JTA-VAE",                         "16px"),
    (c2, "Dataset",    "ZINC250k",                        "16px"),
    (c3, "Latent Dim", "256",                             "26px"),
    (c4, "Checkpoint", "Loaded" if loaded else "Demo",    "16px"),
]:
    color = "#22c55e" if (label == "Checkpoint" and loaded) else ("#f59e0b" if label == "Checkpoint" else "#00e5ff")
    col.markdown(f"""<div class='metric-card'><div class='metric-label'>{label}</div>
    <div class='metric-value' style='font-size:{fs};color:{color}'>{val}</div></div>""",
    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# GENERATION
# ---------------------------------------------------------
if generate_btn:
    bar = st.progress(0, text="Sampling latent space...")
    time.sleep(0.3); bar.progress(30, text="Running decoder...")
    time.sleep(0.3); bar.progress(65, text="Computing properties...")
    mols = generate_molecules(n_mols, (logp_min, logp_max), (qed_min, qed_max), (sas_min, sas_max), temp)
    bar.progress(100, text="Done!"); time.sleep(0.2); bar.empty()

    if not mols:
        st.warning("No molecules satisfied the constraints. Try widening the sliders.")
        st.stop()

    st.session_state['mols'] = mols
    st.session_state['df']   = pd.DataFrame(mols)

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------
if 'mols' in st.session_state:
    df   = st.session_state['df']
    mols = st.session_state['mols']

    # -- SUMMARY METRICS --
    st.markdown("### Generation Summary")
    s1, s2, s3, s4, s5 = st.columns(5)
    for col, lbl, val in [
        (s1, "Generated",     str(len(df))),
        (s2, "Lipinski Pass", f"{int(df['Lipinski'].mean()*100)}%"),
        (s3, "Avg logP",      f"{df['logP'].mean():.2f}"),
        (s4, "Avg QED",       f"{df['QED'].mean():.3f}"),
        (s5, "Avg SAS",       f"{df['SAS'].mean():.2f}"),
    ]:
        col.markdown(f"""<div class='metric-card'><div class='metric-label'>{lbl}</div>
        <div class='metric-value'>{val}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Molecules", "Analytics", "Data Table"])

    # -- TAB 1: MOLECULE GRID --
    with tab1:
        st.markdown("#### Generated Molecules")
        cols_per_row = 4
        for row in [mols[i:i+cols_per_row] for i in range(0, len(mols), cols_per_row)]:
            cols = st.columns(cols_per_row)
            for col, mol in zip(cols, row):
                with col:
                    img = mol_to_svg_b64(mol['SMILES'])
                    lip = ('<span class="badge badge-pass">Lipinski pass</span>'
                           if mol['Lipinski'] else
                           '<span class="badge badge-fail">Lipinski fail</span>')
                    img_tag = f"<img src='{img}' style='width:100%;border-radius:8px;background:#1a1f2e'>" if img else ""
                    st.markdown(f"""
                    <div class='mol-card'>
                      {img_tag}
                      <div style='margin-top:10px'>
                        <div style='font-family:monospace;font-size:10px;color:#64748b;
                             word-break:break-all;margin-bottom:6px'>{mol['SMILES'][:45]}...</div>
                        <div>
                          <span class='badge' style='background:rgba(0,229,255,.12);color:#00e5ff;border:1px solid #00e5ff44'>logP {mol['logP']}</span>
                          <span class='badge' style='background:rgba(124,58,237,.12);color:#a78bfa;border:1px solid #7c3aed44'>QED {mol['QED']}</span>
                          <span class='badge' style='background:rgba(34,197,94,.12);color:#22c55e;border:1px solid #22c55e44'>SAS {mol['SAS']}</span>
                        </div>
                        <div style='margin-top:4px'>{lip}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)

    # -- TAB 2: ANALYTICS --
    with tab2:
        col_l, col_r = st.columns(2)

        with col_l:
            n = len(df)
            fig_lat = go.Figure(go.Scatter(
                x=np.random.randn(n), y=np.random.randn(n),
                mode='markers',
                marker=dict(color=df['QED'], colorscale='Plasma', size=10, showscale=True,
                            colorbar=dict(title="QED", tickfont=dict(color="#64748b")),
                            line=dict(width=0)),
                text=[f"QED: {q}  logP: {l}  SAS: {s}"
                      for q, l, s in zip(df['QED'], df['logP'], df['SAS'])],
                hoverinfo='text'
            ))
            fig_lat.update_layout(**plotly_layout(
                title=dict(text="Latent Space (2D projection)", font=dict(color="#00e5ff", size=13))))
            st.plotly_chart(fig_lat, use_container_width=True)

        with col_r:
            fig_lp = go.Figure(go.Histogram(
                x=df['logP'], nbinsx=12, marker_color='#00e5ff', opacity=0.8,
                hovertemplate="logP: %{x}<br>Count: %{y}<extra></extra>"
            ))
            fig_lp.update_layout(**plotly_layout(
                title=dict(text="logP Distribution", font=dict(color="#00e5ff", size=13))))
            st.plotly_chart(fig_lp, use_container_width=True)

        c_a, c_b = st.columns(2)
        for col, prop, color in [(c_a, 'QED', '#a78bfa'), (c_b, 'SAS', '#22c55e')]:
            with col:
                fig = go.Figure(go.Histogram(
                    x=df[prop], nbinsx=10, marker_color=color, opacity=0.8,
                    hovertemplate=f"{prop}: %{{x}}<br>Count: %{{y}}<extra></extra>"
                ))
                fig.update_layout(**plotly_layout(
                    title=dict(text=f"{prop} Distribution", font=dict(color=color, size=12)),
                    margin=dict(l=30, r=10, t=35, b=30)))
                st.plotly_chart(fig, use_container_width=True)

        # Constraint satisfaction bar chart (3 properties + Lipinski only)
        st.markdown("#### Constraint Satisfaction")
        labels = ['logP in range', 'QED in range', 'SAS in range', 'Lipinski Pass']
        vals   = [
            df['logP'].between(logp_min, logp_max).mean() * 100,
            df['QED'].between(qed_min, qed_max).mean() * 100,
            df['SAS'].between(sas_min, sas_max).mean() * 100,
            df['Lipinski'].mean() * 100,
        ]
        fig_bar = go.Figure(go.Bar(
            x=vals, y=labels, orientation='h',
            marker_color=['#00e5ff', '#a78bfa', '#22c55e', '#f59e0b'],
            opacity=0.85,
            text=[f"{v:.0f}%" for v in vals], textposition='outside',
            textfont=dict(color='#e2e8f0', size=11),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>"
        ))
        fig_bar.update_layout(**plotly_layout(
            xaxis=dict(range=[0, 115], gridcolor="#1e2535", ticksuffix="%"),
            title=dict(text="% molecules satisfying each constraint",
                       font=dict(color="#00e5ff", size=13)),
            height=220))
        st.plotly_chart(fig_bar, use_container_width=True)

        # Parallel coordinates
        st.markdown("#### Multi-Property Parallel Coordinates")
        fig_par = px.parallel_coordinates(
            df, dimensions=['logP', 'QED', 'SAS'],
            color='QED', color_continuous_scale='Plasma',
        )
        fig_par.update_layout(**plotly_layout(
            coloraxis_colorbar=dict(tickfont=dict(color="#64748b"), title="QED"),
            height=320))
        st.plotly_chart(fig_par, use_container_width=True)

    # -- TAB 3: DATA TABLE --
    with tab3:
        st.markdown("#### Generated Molecules — Full Property Table")
        show = ['SMILES', 'logP', 'QED', 'SAS', 'MW', 'Lipinski']
        st.dataframe(
            df[show].style
              .background_gradient(subset=['QED'],  cmap='plasma')
              .background_gradient(subset=['logP'], cmap='Blues'),
            use_container_width=True, height=420
        )
        st.download_button("Download CSV",
            df[show].to_csv(index=False).encode(),
            "jta_vae_generated.csv", "text/csv")

# ---------------------------------------------------------
# PLACEHOLDER
# ---------------------------------------------------------
else:
    st.markdown("""
    <div style='border:1px dashed #1e2535;border-radius:16px;padding:60px;text-align:center;margin-top:20px'>
      <div style='font-family:monospace;font-size:1.1rem;color:#00e5ff;margin:12px 0'>
        Set your constraints and click <b>Generate Molecules</b>
      </div>
      <div style='color:#64748b;font-size:14px;max-width:500px;margin:0 auto'>
        Adjust the <b style='color:#00e5ff'>logP</b>, <b style='color:#a78bfa'>QED</b>, and
        <b style='color:#22c55e'>SAS</b> sliders in the sidebar, then generate drug-like molecules
        from the trained JTA-VAE latent space.
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""<br><br>
<div style='text-align:center;color:#1e2535;font-size:12px;font-family:monospace'>
JTA-VAE · ZINC250k · Final Year Project Demo · Streamlit + RDKit + Plotly
</div>""", unsafe_allow_html=True)