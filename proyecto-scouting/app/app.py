# app/app.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ===================== Config / Head =====================
st.set_page_config(page_title="Scouting LaLiga", layout="wide")

st.markdown("""
<style>
.big-title {font-size:2.1rem; font-weight:800; margin:0 0 .25rem 0;}
.subtle {color:#8A8F98; margin:0 0 1.0rem 0;}
.kpi .stMetric {text-align:center}
</style>
<div class="big-title"> Scouting Hub — Radar de rendimiento de las 5 grandes ligas</div>
<p class="subtle">Análisis operativo para dirección deportiva: jugadores con ≥900′, métricas por 90’ y porcentajes (0–100). Filtros por competición, <b>rol táctico</b> y temporada.</p>
""", unsafe_allow_html=True)

# ===================== Carga de datos =====================
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PARQUET = next((DATA_DIR.glob("scouting_laliga_df_final_*.parquet")), None)
CSV_FALLBACK = next((DATA_DIR.glob("scouting_laliga_df_final_*_fallback.csv")), None)

@st.cache_data
def load_data():
    if PARQUET and PARQUET.exists():
        try:
            return pd.read_parquet(PARQUET)
        except Exception:
            pass
    if CSV_FALLBACK and CSV_FALLBACK.exists():
        return pd.read_csv(CSV_FALLBACK)
    st.error(f"No encuentro el dataset en {DATA_DIR}")
    return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# ===================== Helpers ===========================
def normalize_0_1(df_num: pd.DataFrame) -> pd.DataFrame:
    mn = df_num.min(axis=0)
    mx = df_num.max(axis=0)
    return (df_num - mn) / (mx - mn + 1e-9)

# ===================== Query params (nueva API) ==========
params = dict(st.query_params)
def _to_list(v): return [] if v is None else (v if isinstance(v, list) else [v])

comp_pre   = _to_list(params.get("comp"))
rol_pre    = _to_list(params.get("rol"))          # Rol_Tactico
season_pre = _to_list(params.get("season"))
min_pre    = int(params.get("min", 900))
age_from   = int(params.get("age_from", 15))
age_to     = int(params.get("age_to", 40))

# ===================== Filtros ===========================
st.sidebar.header("Filtros")

comp_opts   = sorted(df["Comp"].dropna().unique())
rol_opts    = sorted(df["Rol_Tactico"].dropna().unique())
season_opts = sorted(df["Season"].dropna().unique())

comp   = st.sidebar.multiselect("Competición", comp_opts, default=comp_pre)
rol    = st.sidebar.multiselect("Rol táctico (posición funcional)", rol_opts, default=rol_pre)
season = st.sidebar.multiselect("Temporada", season_opts, default=season_pre)

# Minutos: mínimo 900 (no se permite bajar de ahí)
global_min = max(900, int(df["Min"].min())) if "Min" in df else 900
global_max = int(df["Min"].max()) if "Min" in df else 3420
default_min = int(np.clip(900, global_min, global_max))

min_sel_slider = st.sidebar.slider("Minutos jugados (≥)", min_value=global_min, max_value=global_max,
                                   value=default_min, key="mins_slider")
min_sel = st.sidebar.number_input("Escribir minutos (≥)", min_value=global_min, max_value=global_max,
                                  value=int(min_sel_slider), step=30, key="mins_num")

# Edad
age_num = pd.to_numeric(df.get("Age", pd.Series(dtype=float)), errors="coerce")
if age_num.size:
    age_min, age_max = int(np.nanmin(age_num)), int(np.nanmax(age_num))
else:
    age_min, age_max = 15, 40
age_default = (max(age_min, age_from), min(age_max, age_to))
age_range_slider = st.sidebar.slider("Edad (rango)", min_value=age_min, max_value=age_max,
                                     value=age_default, key="age_slider")
age_min_num = st.sidebar.number_input("Edad mínima", min_value=age_min, max_value=age_max,
                                      value=int(age_range_slider[0]), step=1, key="age_min_num")
age_max_num = st.sidebar.number_input("Edad máxima", min_value=age_min, max_value=age_max,
                                      value=int(age_range_slider[1]), step=1, key="age_max_num")
age_range = (int(min(age_min_num, age_max_num)), int(max(age_min_num, age_max_num)))

# Aplica filtros
mask = (df["Min"] >= min_sel) if "Min" in df else True
if age_num.size: mask &= age_num.between(age_range[0], age_range[1])
if comp:   mask &= df["Comp"].isin(comp)
if rol:    mask &= df["Rol_Tactico"].isin(rol)
if season: mask &= df["Season"].isin(season)

dff = df.loc[mask].copy()

# Escribe estado en URL
st.query_params.update({
    "comp": comp, "rol": rol, "season": season,
    "min": str(min_sel),
    "age_from": str(age_range[0]), "age_to": str(age_range[1]),
})

# ===================== Métricas ===================================================
gk_metrics = ["Save%", "PSxG+/-_per90", "PSxG_per90", "Saves_per90", "CS%", "Launch%"]
out_metrics = [
    "Gls_per90","xG_per90","NPxG_per90","Sh_per90","SoT_per90","G/SoT_per90",
    "xA_per90","KP_per90","GCA90_per90","SCA_per90",
    "PrgP_per90","PrgC_per90","Carries_per90",
    "Cmp%","Cmp_per90","Tkl+Int_per90","Int_per90","Recov_per90"
]
metrics_all = [m for m in out_metrics if m in dff.columns]

# ===================== Tabs ===============================================
tab_overview, tab_ranking, tab_compare, tab_similarity, tab_shortlist = st.tabs(
    ["📊 Overview", "🏆 Ranking", "🆚 Comparador", "🧬 Similares", "⭐ Shortlist"]
)

# ===================== Mensaje si no hay jugadores =========================
def stop_if_empty(dfx):
    if len(dfx) == 0:
        st.warning("No hay jugadores que cumplan con estas condiciones de filtro. "
                   "Prueba a bajar el umbral de minutos, ampliar las edades o seleccionar más roles/temporadas.")
        st.stop()

# ===================== OVERVIEW ===========================================
with tab_overview:
    stop_if_empty(dff)

    # KPIs sobre el subconjunto filtrado
    k1, k2, k3, k4 = st.columns(4, gap="large")
    with k1: st.metric("Jugadores (en filtro)", f"{len(dff):,}")
    with k2: st.metric("Equipos (en filtro)", f"{dff['Squad'].nunique()}")
    with k3:
        try:
            st.metric("Media de edad (en filtro)", f"{pd.to_numeric(dff['Age'], errors='coerce').mean():.1f}")
        except Exception:
            st.metric("Media de edad (en filtro)", "—")
    with k4:
        med = int(dff["Min"].median()) if "Min" in dff and len(dff) else 0
        st.metric("Minutos medianos (en filtro)", f"{med:,}")

    # ===== Productividad ofensiva =====
    st.markdown("### Productividad ofensiva: **xG/90 vs Goles/90**")
    if all(c in dff.columns for c in ["xG_per90","Gls_per90"]):
        fig = px.scatter(
            dff, x="xG_per90", y="Gls_per90",
            color="Rol_Tactico", size=dff.get("SoT_per90", None),
            hover_name="Player",
            labels={"xG_per90":"xG por 90", "Gls_per90":"Goles por 90", "Rol_Tactico":"Rol táctico", "SoT_per90":"Tiros a puerta/90"},
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Faltan columnas xG_per90 / Gls_per90 en el subconjunto actual.")

    # ===== Creación de juego =====
    st.markdown("### Creación: **xA/90 vs Pases clave/90** (tamaño = GCA/90)")
    if all(c in dff.columns for c in ["xA_per90","KP_per90","GCA90_per90"]):
        fig = px.scatter(
            dff, x="xA_per90", y="KP_per90",
            size="GCA90_per90", color="Rol_Tactico", hover_name="Player",
            labels={"xA_per90":"xA por 90", "KP_per90":"Pases clave por 90", "GCA90_per90":"Acciones que generan gol/90", "Rol_Tactico":"Rol táctico"},
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ===== Progresión (Top 15) =====
    st.markdown("### Progresión: **Top 15** en Pases progresivos por 90")
    if "PrgP_per90" in dff.columns:
        top_prog = dff.sort_values("PrgP_per90", ascending=False).head(15)
        fig = px.bar(
            top_prog.sort_values("PrgP_per90"),
            x="PrgP_per90", y="Player", color="Rol_Tactico",
            labels={"PrgP_per90":"Pases progresivos por 90", "Player":"Jugador", "Rol_Tactico":"Rol táctico"},
            template="plotly_dark", orientation="h"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ===== Defensa =====
    st.markdown("### Defensa: **Tkl+Int/90 vs Recuperaciones/90** (tamaño = Intercepciones/90)")
    if all(c in dff.columns for c in ["Tkl+Int_per90","Recov_per90","Int_per90"]):
        fig = px.scatter(
            dff, x="Tkl+Int_per90", y="Recov_per90", size="Int_per90",
            color="Rol_Tactico", hover_name="Player",
            labels={"Tkl+Int_per90":"Entradas + Intercepciones por 90", "Recov_per90":"Recuperaciones por 90", "Int_per90":"Intercepciones por 90", "Rol_Tactico":"Rol táctico"},
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ===== Pase =====
    st.markdown("### Pase: **Precisión** vs **Volumen**")
    if all(c in dff.columns for c in ["Cmp%","Cmp_per90"]):
        fig = px.scatter(
            dff, x="Cmp%", y="Cmp_per90", color="Rol_Tactico", hover_name="Player",
            labels={"Cmp%":"Precisión de pase (%)", "Cmp_per90":"Pases completados por 90", "Rol_Tactico":"Rol táctico"},
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ===== Porteros (si los hay) =====
    if "Save%" in dff.columns and "PSxG+/-_per90" in dff.columns and "Saves_per90" in dff.columns:
        # filtra si hay roles portero en Rol_Tactico
        gk_df = dff[dff["Rol_Tactico"].str.contains("GK|Portero", case=False, na=False)].copy()
        if len(gk_df):
            st.markdown("### Porteros: **% Paradas** vs **PSxG+/- por 90** (tamaño = Paradas/90)")
            fig = px.scatter(
                gk_df, x="Save%", y="PSxG+/-_per90", size="Saves_per90",
                hover_name="Player", color="Rol_Tactico",
                labels={"Save%":"% Paradas", "PSxG+/-_per90":"PSxG +/- por 90", "Saves_per90":"Paradas por 90", "Rol_Tactico":"Rol táctico"},
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

# ===================== RANKING ===========================================
with tab_ranking:
    stop_if_empty(dff)
    st.subheader("Ranking por métrica")
    metric_to_rank = st.selectbox("Métrica para ordenar", metrics_all, index=0 if metrics_all else None)
    topn = st.slider("Top N", 5, 100, 20)
    cols_show = ["Player","Squad","Season","Rol_Tactico","Comp","Min","Age"] + metrics_all
    tabla = dff[cols_show].sort_values(metric_to_rank, ascending=False).head(topn)
    st.dataframe(tabla, use_container_width=True)

# ===================== COMPARADOR ========================================
with tab_compare:
    stop_if_empty(dff)
    st.subheader("Comparador de jugadores (Radar)")
    players = dff["Player"].dropna().unique().tolist()
    cA, cB = st.columns(2)
    p1 = cA.selectbox("Jugador A", players, index=0 if players else None, key="pA")
    p2 = cB.selectbox("Jugador B", players, index=1 if len(players)>1 else 0, key="pB")

    radar_feats = st.multiselect("Métricas para el radar (elige 4–8)", metrics_all, default=metrics_all[:6], key="feats")

    def radar(df_in, pA, pB, feats):
        S = df_in[feats].astype(float)
        S = normalize_0_1(S)
        A = S[df_in["Player"]==pA].mean(numeric_only=True).fillna(0)
        B = S[df_in["Player"]==pB].mean(numeric_only=True).fillna(0)
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=A.values, theta=feats, fill="toself", name=pA))
        fig.add_trace(go.Scatterpolar(r=B.values, theta=feats, fill="toself", name=pB))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True)
        return fig

    if p1 and p2 and radar_feats:
        st.plotly_chart(radar(dff, p1, p2, radar_feats), use_container_width=True)

# ===================== SIMILARES =========================================
with tab_similarity:
    stop_if_empty(dff)
    st.subheader("Jugadores similares (cosine similarity)")
    feats_sim = st.multiselect("Selecciona 6–12 métricas", metrics_all, default=metrics_all[:8], key="sim_feats")
    target = st.selectbox("Jugador objetivo", dff["Player"].dropna().unique().tolist())
    if feats_sim and target:
        X = dff[feats_sim].astype(float).fillna(0.0).to_numpy()
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-9)
        from numpy.linalg import norm
        idx = dff.index[dff["Player"]==target][0]
        v = X[dff.index.get_loc(idx)]
        sims = (X @ v) / (norm(X, axis=1)*norm(v) + 1e-9)
        out = dff[["Player","Squad","Season","Rol_Tactico","Comp","Min","Age"]].copy()
        out["similarity"] = sims
        st.dataframe(out.sort_values("similarity", ascending=False).head(25), use_container_width=True)

# ===================== SHORTLIST =========================================
with tab_shortlist:
    stop_if_empty(dff)
    st.subheader("Shortlist (lista de seguimiento)")
    if "shortlist" not in st.session_state: st.session_state.shortlist = []
    to_add = st.multiselect("Añadir jugadores", dff["Player"].dropna().unique().tolist())
    if st.button("➕ Agregar seleccionados"): st.session_state.shortlist = sorted(set(st.session_state.shortlist) | set(to_add))
    to_remove = st.multiselect("Eliminar de shortlist", st.session_state.shortlist)
    if st.button("🗑️ Eliminar seleccionados"): st.session_state.shortlist = [p for p in st.session_state.shortlist if p not in set(to_remove)]
    sh = dff[dff["Player"].isin(st.session_state.shortlist)]
    st.dataframe(sh, use_container_width=True)
    st.download_button("⬇️ Descargar shortlist (CSV)", data=sh.to_csv(index=False).encode("utf-8-sig"),
                       file_name="shortlist_scouting.csv", mime="text/csv")

# ===================== Footer trazabilidad ===============
meta = next(DATA_DIR.glob("metadata_*.json"), None)
if meta and meta.exists():
    import json
    m = json.loads(meta.read_text(encoding="utf-8"))
    st.caption(f"📦 Dataset: {m.get('files',{}).get('parquet','parquet')} · "
               f"Filtros base: ≥{m.get('filters',{}).get('minutes_min',900)}′ · "
               f"Generado: {m.get('created_at','')}")

