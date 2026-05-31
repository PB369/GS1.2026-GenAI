"""
app/pages/prediction.py
-----------------------
Responsabilidade: renderizar a página de predição de risco em tempo real.
Exibe as probabilidades de cada modelo e o waterfall SHAP por classe
para o ponto definido pelos sliders.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.styles import section_hdr, info_box, plotly_cfg, base_layout, CLS_COLOR


def render(D: dict, X_input_sel: np.ndarray, shap_point: np.ndarray,
           shap_point_mean: np.ndarray, prob_xgb: np.ndarray,
           prob_rf: np.ndarray, label_xgb: str, label_rf: str) -> None:

    section_hdr("Predição de Risco em Tempo Real")

    # ── Probabilidades lado a lado
    col_a, col_b = st.columns(2)
    for col, name, probs, lbl in [
        (col_a, "XGBoost - Melhor modelo com os parâmetros nos valores padrões", prob_xgb, label_xgb),
        (col_b, "Random Forest - Exibido para fins de comparação",   prob_rf,  label_rf),
    ]:
        with col:
            st.markdown(f"**{name}**")
            fig = go.Figure()
            for cls, p in zip(D["CLASSES"], probs):
                fig.add_trace(go.Bar(
                    x=[p * 100], y=[cls],
                    orientation="h",
                    marker_color=CLS_COLOR[cls],
                    text=f"{p*100:.1f}%",
                    textposition="outside",
                    name=cls,
                ))
            fig.update_layout(**base_layout(
                showlegend=False,
                xaxis_title="Probabilidade (%)",
                xaxis_range=[0, 115],
                height=230,
            ))
            st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

    st.markdown("---")

    # ── Waterfall SHAP por classe 
    section_hdr("Contribuição SHAP no XGBoost")
    st.caption("Mostra exatamente como o modelo chegou à predição baseada nos valores atuais dos parêmtros.")

    tabs = st.tabs([f"Classe: {cls}" for cls in D["CLASSES"]])
    for t, cls_idx in zip(tabs, range(3)):
        with t:
            vals  = shap_point[0, :, cls_idx]
            order = np.argsort(np.abs(vals))[::-1]
            feat_names = D["SEL_FEAT"]

            fig_wf = go.Figure(go.Bar(
                x=vals[order],
                y=[feat_names[i] for i in order],
                orientation="h",
                marker_color=["#ef4444" if v > 0 else "#22c55e" for v in vals[order]],
                text=[f"{v:+.3f}" for v in vals[order]],
                textposition="outside",
            ))
            fig_wf.add_vline(x=0, line_color="gray", line_width=1)
            fig_wf.update_layout(**base_layout(
                title=f"Classe '{D['CLASSES'][cls_idx]}' — prob atual: {prob_xgb[cls_idx]*100:.1f}%",
                xaxis_title="SHAP (+ aumenta prob desta classe, − reduz)",
                height=340,
                margin=dict(l=10, r=70, t=50, b=10),
            ))
            st.plotly_chart(fig_wf, use_container_width=True, config=plotly_cfg())

    info_box(
        "🔴 Vermelho = feature aumenta o risco para esta classe &nbsp;|&nbsp; "
        "🟢 Verde = feature reduz o risco &nbsp;",
    )
