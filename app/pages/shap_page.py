"""
app/pages/shap_page.py
----------------------
Responsabilidade: renderizar a página de Interpretabilidade SHAP.
Combina visão global (conjunto de teste fixo) com visão local
(ponto atual dos sliders).
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.styles import section_hdr, info_box, plotly_cfg, base_layout

def render(D: dict, X_input_sel: np.ndarray, shap_point: np.ndarray,
           shap_point_mean: np.ndarray, prob_xgb: np.ndarray) -> None:

    sv  = D["sv_xgb"]
    SEL = D["SEL_FEAT"]
    mean_abs = np.abs(sv).mean(axis=(0, 2))

    # ── Bloco 1: comparativo de SHAP
    section_hdr("SHAP pré-gerado vs. SHAP atual")
    st.caption("Esquerda: média sobre 400 amostras | Direita: SHAP do ponto atual dos sliders.")

    col1, col2 = st.columns(2)
    with col1:
        order_g   = np.argsort(mean_abs)
        colors_g  = ["#4ade80" if mean_abs[i] == mean_abs.max() else "#1e6e42" for i in order_g]
        fig_global = go.Figure(go.Bar(
            x=mean_abs[order_g],
            y=[SEL[i] for i in order_g],
            orientation="h",
            marker_color=colors_g,
            text=[f"{v:.3f}" for v in mean_abs[order_g]],
            textposition="outside",
        ))

        fig_global.update_layout(**base_layout(
            title="Configuração Padrão",
            xaxis_title="SHAP médio",
            height=370, margin=dict(l=10, r=90, t=50, b=10),
        ))
        st.plotly_chart(fig_global, use_container_width=True, config=plotly_cfg())

    with col2:
        order_p  = np.argsort(np.abs(shap_point_mean))
        colors_p = ["#ef4444" if v > 0 else "#22c55e" for v in shap_point_mean[order_p]]
        fig_pt = go.Figure(go.Bar(
            x=shap_point_mean[order_p],
            y=[SEL[i] for i in order_p],
            orientation="h",
            marker_color=colors_p,
            text=[f"{v:+.3f}" for v in shap_point_mean[order_p]],
            textposition="outside",
        ))
        fig_pt.update_layout(**base_layout(
            title="Configuração Atual",
            xaxis_title="SHAP médio",
            height=370, margin=dict(l=10, r=70, t=50, b=10),
        ))
        st.plotly_chart(fig_pt, use_container_width=True, config=plotly_cfg())

    st.markdown("---")

    # ── Bloco 2: Scatter SHAP × valor da feature 
    section_hdr("SHAP × valor da feature")
    st.caption("Pontos coloridos: 400 amostras de predição padrão. Ícone ★: posição da configuração atual dos sliders.")

    feat_sel = st.selectbox("Feature selecionada:", SEL,
                            index=SEL.index("temp_superficie") if "temp_superficie" in SEL else 0,
                            key="shap_scatter_feat")
    fi = SEL.index(feat_sel)
    fig_sc = go.Figure(go.Scatter(
        x=D["Xte"][:, fi],
        y=np.abs(sv[:, fi, :]).mean(axis=1),
        mode="markers",
        marker=dict(color=D["Xte"][:, fi], colorscale="RdYlGn_r",
                    size=5, opacity=0.55, colorbar=dict(title="")),
    ))
    fig_sc.add_trace(go.Scatter(
        x=[X_input_sel[0, fi]], y=[float(shap_point_mean[fi])],
        marker=dict(color="white", size=14, symbol="star",
                    line=dict(color="#4ade80", width=2)),
        mode="markers", name="★ Ponto atual",
    ))
    fig_sc.update_layout(**base_layout(
        title=f"SHAP × {feat_sel}  |  ★ move com o slider",
        xaxis_title=f"{feat_sel} (normalizado)",
        yaxis_title="|SHAP| médio", height=350,
    ))
    st.plotly_chart(fig_sc, use_container_width=True, config=plotly_cfg())

    st.markdown("---")

    # ── Bloco 3: Heatmap por classe com coluna do ponto atual ─────────────────
    section_hdr("Heatmap SHAP - Amostras + Configuração Atual")
    st.caption("A última coluna 'Atual' representa o valor atual dos sliders, equanto que as demais representa as amostras pré-testadas.")

    cls_tab = st.tabs([f"Classe {c}" for c in D["CLASSES"]])
    for t, ci in zip(cls_tab, range(3)):
        with t:
            idx_s   = np.random.default_rng(42).choice(sv.shape[0], size=80, replace=False)
            heat    = sv[idx_s, :, ci].T
            pt_col  = shap_point[0, :, ci].reshape(-1, 1)
            combined = np.hstack([heat, pt_col])
            x_labels = [f"s{i}" for i in range(80)] + ["Atual"]

            fig_h = go.Figure(go.Heatmap(
                z=combined, x=x_labels, y=SEL,
                colorscale="RdYlGn_r", zmid=0,
                colorbar=dict(title="SHAP"),
            ))
            fig_h.add_vline(
                x=len(x_labels) - 1.5,
                line_dash="dash", line_color="black", line_width=1.8,
            )
            fig_h.update_layout(**base_layout(
                title=f"Classe '{D['CLASSES'][ci]}'  |  Última coluna = ponto dos sliders",
                height=400,
            ))
            st.plotly_chart(fig_h, use_container_width=True, config=plotly_cfg())
