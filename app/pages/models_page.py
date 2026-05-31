"""
app/pages/models_page.py
------------------------
Responsabilidade: renderizar a página de Comparação de Modelos.
Métricas globais fixas + indicadores dinâmicos do ponto dos sliders.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from app.styles import section_hdr, info_box, plotly_cfg, base_layout, CLS_COLOR

def render(D: dict, prob_xgb: np.ndarray, prob_rf: np.ndarray,
           label_xgb: str, label_rf: str) -> None:

    m       = D["metrics"]
    CLASSES = D["CLASSES"]

    # ── KPIs 
    section_hdr("Desempenho global — avaliado sobre 400 amostras de teste previamente realizado")
    st.caption("Apresentação das métricas de acurácia geral e de acurácia média de Cross Validation (CV) para os modelos de Random Forest (RF) e XGBoost (XGB).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RF — Acurácia",  f"{m['Random Forest']['acc']*100:.2f}%")
    c2.metric("RF — CV Média",  f"{m['Random Forest']['cv_mean']*100:.2f}%",
              f"±{m['Random Forest']['cv_std']*100:.2f}%")
    c3.metric("XGB — Acurácia", f"{m['XGBoost']['acc']*100:.2f}%", delta="melhor ✓")
    c4.metric("XGB — CV Média", f"{m['XGBoost']['cv_mean']*100:.2f}%",
              f"±{m['XGBoost']['cv_std']*100:.2f}%")

    st.markdown("---")

    # ── Matrizes de confusão com destaque da classe atual 
    section_hdr("Matrizes de confusão")
    st.caption(f" A borda colorida representa a classe prevista para o ponto atual.")

    col_a, col_b = st.columns(2)
    for col, mname, pred_lbl in zip(
        [col_a, col_b], ["Random Forest", "XGBoost"], [label_rf, label_xgb]
    ):
        with col:
            cm = m[mname]["cm"]
            pred_idx = CLASSES.index(pred_lbl)
            fig_cm = go.Figure(go.Heatmap(
                z=cm[::-1], x=CLASSES, y=CLASSES[::-1],
                text=cm[::-1].astype(str), texttemplate="%{text}",
                colorscale=[[0, "#0d1f1a"], [1, "#4ade80"]],
                showscale=False,
            ))
            fig_cm.add_shape(
                type="rect",
                x0=pred_idx - 0.5, x1=pred_idx + 0.5,
                y0=-0.5, y1=2.5,
                line=dict(color=CLS_COLOR[pred_lbl], width=3),
                fillcolor="rgba(0,0,0,0)",
            )
            fig_cm.add_annotation(
                x=pred_idx, y=2.7,
                text="Classe de Risco Atual", showarrow=False,
                font=dict(color=CLS_COLOR[pred_lbl], size=10),
            )
            fig_cm.update_layout(**base_layout(
                title=f"{mname}  |  Ponto atual → <b>{pred_lbl}</b>",
                xaxis_title="Previsto", yaxis_title="Real",
                height=330, margin=dict(l=10, r=10, t=65, b=10),
            ))
            st.plotly_chart(fig_cm, use_container_width=True, config=plotly_cfg())

    st.markdown("---")

    # ── F1 por classe
    section_hdr("F1-score por classe do dataset pré-testado")
    st.caption("As barras azuis representam o Random Forest e as barras vermelhas o XGBoost")

    fig_f1 = go.Figure()
    for mname in ["Random Forest", "XGBoost"]:
        f1s = [m[mname]["report"][cls]["f1-score"] for cls in CLASSES]
        fig_f1.add_trace(go.Bar(
            x=CLASSES, y=f1s, name=mname,
            text=[f"{v:.3f}" for v in f1s], textposition="outside",
        ))
    
    fig_f1.update_layout(**base_layout(
        barmode="group",
        title="F1-score por classe de risco",
        yaxis_range=[0, 1.25], yaxis_title="F1-score",
        height=350,
    ))
    st.plotly_chart(fig_f1, use_container_width=True, config=plotly_cfg())

    st.markdown("---")

    # ── Probabilidades do ponto atual
    section_hdr("Probabilidades das classes para o ponto atual — atualiza com cada slider")
    st.caption("Exibe a probabilidade de escolha das classes de risco por cada modelo, baseado na configuração atual dos parâmetros definidos nos sliders.")

    fig_cmp = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"XGBoost  →  {label_xgb}", f"Random Forest  →  {label_rf}"],
    )
    for ci, (probs, lbl) in enumerate([(prob_xgb, label_xgb), (prob_rf, label_rf)], 1):
        bar_colors = [
            CLS_COLOR[c] if c == lbl
            else f"rgba({','.join(str(int(CLS_COLOR[c][1:][i:i+2], 16)) for i in (0, 2, 4))},0.35)"
            for c in CLASSES
        ]
        fig_cmp.add_trace(
            go.Bar(x=CLASSES, y=[p * 100 for p in probs],
                   marker_color=bar_colors,
                   text=[f"{p*100:.1f}%" for p in probs],
                   textposition="outside", showlegend=False),
            row=1, col=ci,
        )
    fig_cmp.update_yaxes(range=[0, 115], title_text="Probabilidade (%)")
    fig_cmp.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=330, margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig_cmp, use_container_width=True, config=plotly_cfg())