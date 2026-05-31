"""
app/pages/eda.py
----------------
Responsabilidade: renderizar a página de Análise Exploratória.
Base fixa do dataset + marcadores dinâmicos do ponto dos sliders.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.styles import section_hdr, info_box, plotly_cfg, base_layout, CLS_COLOR


def render(D: dict, X_input_sel: np.ndarray, slider_vals: dict,
           label_xgb: str, temp: float, ndvi: float) -> None:

    df_raw = D["df_raw"]

    # ── Bloco 1: Distribuição de classes + scatter Temp × NDVI
    section_hdr("Visão geral do dataset pré-utilizado")
    st.caption("A distribuição das classes não muda com os sliders, pois ela reflete o dataset completo pré-gerado pela IA. O ícone ★ mostra onde a configuração atual dos parâmetros se posiciona em relação ao dataset.")

    col1, col2 = st.columns(2)
    with col1:
        counts = df_raw["risco_desmatamento"].value_counts()
        fig = go.Figure(go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=[CLS_COLOR[c] for c in counts.index],
            text=[f"{v}<br>({v/len(df_raw)*100:.1f}%)" for v in counts.values],
            textposition="outside",
        ))
        fig.add_annotation(
            x=label_xgb, y=counts[label_xgb],
            text="Predição atual", showarrow=False,
            font=dict(color="white", size=11), yshift=45,
        )
        fig.update_layout(**base_layout(
            title=f"Distribuição das Classes  |  Predição atual → <b>{label_xgb}</b>",
            yaxis_title="Quantidade", height=310,
        ))
        st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

    with col2:
        fig2 = go.Figure()
        for cls in D["CLASSES"]:
            sub = df_raw[df_raw["risco_desmatamento"] == cls]
            fig2.add_trace(go.Scatter(
                x=sub["temp_superficie"], y=sub["ndvi"],
                mode="markers", name=cls,
                marker=dict(color=CLS_COLOR[cls], size=4, opacity=0.35),
            ))
        fig2.add_trace(go.Scatter(
            x=[temp], y=[ndvi],
            mode="markers", name=f"Ponto atual ({label_xgb})",
            marker=dict(color="white", size=12, symbol="star",
                        line=dict(color="#4ade80", width=1)),
        ))
        fig2.update_layout(**base_layout(
            title="Temperatura × NDVI — ★ move com os sliders",
            xaxis_title="Temperatura superficial (°C)",
            yaxis_title="NDVI", height=310,
        ))
        st.plotly_chart(fig2, use_container_width=True, config=plotly_cfg())

    st.markdown("---")

    # ── Bloco 2: Violin por feature 
    section_hdr("Distribuição por feature e classe")
    st.caption("A forma do violino é fixa (dataset). A linha tracejada representa o valor atual de temp_superficie em relação ao dataset.")

    feat_sel    = st.selectbox("Feature para visualizar:", D["SEL_FEAT"],
                               index=D["SEL_FEAT"].index("temp_superficie")
                               if "temp_superficie" in D["SEL_FEAT"] else 0)
    current_val = slider_vals.get(feat_sel, float("nan"))

    fig3 = go.Figure()
    for cls in D["CLASSES"]:
        vals_cls = df_raw[df_raw["risco_desmatamento"] == cls][feat_sel].dropna()
        fig3.add_trace(go.Violin(
            x=[cls] * len(vals_cls), y=vals_cls,
            name=cls, fillcolor=CLS_COLOR[cls],
            line_color=CLS_COLOR[cls], opacity=0.65,
            box_visible=True, meanline_visible=True,
        ))
    if not np.isnan(current_val):
        fig3.add_hline(
            y=current_val,
            line_dash="dash", line_color="white", line_width=2,
            annotation_text=f"Slider: {current_val:.2f}",
            annotation_position="top right",
            annotation_font=dict(color="white", size=12),
        )
    fig3.update_layout(**base_layout(
        title=f"{feat_sel}  |  valor atual = {current_val:.2f}",
        yaxis_title=feat_sel, height=370,
        margin=dict(l=10, r=10, t=55, b=10),
        violinmode="group",
    ))
    st.plotly_chart(fig3, use_container_width=True, config=plotly_cfg())

    # Percentis
    pct_cols = st.columns(3)
    for col, cls in zip(pct_cols, D["CLASSES"]):
        vals_cls = df_raw[df_raw["risco_desmatamento"] == cls][feat_sel].dropna()
        pct = float(np.mean(vals_cls <= current_val) * 100)
        col.metric(f"Percentil em {cls}", f"{pct:.0f}º",
                   help=f"{current_val:.2f} está acima de {pct:.0f}% dos registros de '{cls}'")

    st.markdown("---")

    # ── Bloco 3: Correlação de Pearson 
    section_hdr("Correlação de Pearson — features × target (dataset completo)")
    st.caption("A barra amarela indica a feature onde a predição atual mais se afasta da média do dataset pré-utilizado.")

    pearson = D["pearson"]
    feat_means = pd.DataFrame(D["Xsel"], columns=D["SEL_FEAT"]).mean()
    feat_stds  = pd.DataFrame(D["Xsel"], columns=D["SEL_FEAT"]).std()
    point_zscores = {
        f: abs((X_input_sel[0, D["SEL_FEAT"].index(f)] - feat_means[f]) / (feat_stds[f] + 1e-9))
        for f in D["SEL_FEAT"]
    }
    most_deviant = max(point_zscores, key=point_zscores.get)

    bar_colors = [
        "#facc15" if feat == most_deviant
        else ("#ef4444" if pearson[feat] < 0 else "#22c55e")
        for feat in pearson.index
    ]
    fig4 = go.Figure(go.Bar(
        x=pearson.index, y=pearson.values,
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in pearson.values],
        textposition="outside",
    ))
    fig4.add_hline(y=0, line_color="gray", line_width=0.8)
    fig4.add_annotation(
        x=most_deviant, y=pearson[most_deviant],
        text=f"Maior desvio<br>z={point_zscores[most_deviant]:.1f}",
        showarrow=True, arrowhead=2,
        font=dict(color="#facc15", size=10), arrowcolor="#facc15",
        yshift=20 if pearson[most_deviant] > 0 else -50,
    )
    fig4.update_layout(**base_layout(
        title="Correlação de Pearson  |  🟡 = feature mais desviante do ponto atual",
        yaxis_title="Correlação", height=330,
        margin=dict(l=10, r=10, t=55, b=10),
    ))
    st.plotly_chart(fig4, use_container_width=True, config=plotly_cfg())
