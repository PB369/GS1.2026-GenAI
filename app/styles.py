"""
app/styles.py
-------------
Responsabilidade: centralizar estilos CSS, paleta de cores e
configurações reutilizadas por todas as páginas do app.
"""

import streamlit as st

# ── Paleta 
CLS_COLOR = {
    "Alto":  "#ef4444",
    "Médio": "#f59e0b",
    "Baixo": "#22c55e",
}

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_PAPER_BG = "rgba(0,0,0,0)"
PLOTLY_PLOT_BG  = "rgba(0,0,0,0)"


# ── Helpers 
def plotly_cfg() -> dict:
    """Configuração padrão para ocultar a barra de ferramentas do Plotly."""
    return {"displayModeBar": False}


def base_layout(**kwargs) -> dict:
    """Layout base compartilhado entre todos os gráficos Plotly."""
    return {
        "template":     PLOTLY_TEMPLATE,
        "paper_bgcolor": PLOTLY_PAPER_BG,
        "plot_bgcolor":  PLOTLY_PLOT_BG,
        "margin": dict(l=10, r=10, t=50, b=10),
        **kwargs,
    }


# ── CSS global 
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.hero {
    background: #0d1f1a;
    padding: 1.8rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    border: 1px solid #1e4035;
}
.hero h1 { color: #4ade80; font-size: 2rem; margin: 0; font-weight: 600; letter-spacing: -0.5px; }
.hero p  { color: #86a898; font-size: 0.9rem; margin: 0.4rem 0 0; }

.kpi { background: #0d1f1a; border: 1px solid #1e4035; border-radius: 10px; padding: 1rem 1.2rem; }
.kpi .val { font-size: 1.6rem; font-weight: 600; color: #4ade80; font-family: 'IBM Plex Mono', monospace; }
.kpi .lbl { font-size: 0.78rem; color: #86a898; margin-top: 2px; }

.section-hdr {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #86a898;
    border-left: 3px solid #4ade80; padding-left: 0.6rem; margin-bottom: 1rem;
}

.info-box {
    background: #0d1f1a; border: 1px solid #1e4035; border-radius: 8px;
    padding: 0.8rem 1rem; font-size: 0.82rem; color: #86a898; margin-top: 0.5rem;
}
</style>
"""


def inject_css() -> None:
    """Injeta o CSS global no app Streamlit."""
    st.markdown(CSS, unsafe_allow_html=True)


def section_hdr(text: str) -> None:
    st.markdown(f'<div class="section-hdr">{text}</div>', unsafe_allow_html=True)


def info_box(text: str) -> None:
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)
