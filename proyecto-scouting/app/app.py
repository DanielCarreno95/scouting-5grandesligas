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
/* Títulos principales */
.big-title {font-size:2.1rem; font-weight:800; margin:0 0 .25rem 0;}
.subtle {color:#8A8F98; margin:0 0 1.0rem 0;}
.kpi .stMetric {text-align:center}

/* Sidebar: tipografía y separación */
div[data-testid="stSidebar"] * { font-size: 0.95rem; }
div[data-testid="stSidebar"] label { font-weight: 600; }
div[data-testid="stSidebar"] .stMultiSelect, 
div[data-testid="stSidebar"] .stSlider,
div[data-testid="stSidebar"] .stRadio { margin-bottom: .75rem; }

/* Notas/leyendas pequeñas */
.note {font-size: .85rem; color:#9aa2ad; line-height:1.25rem;}
.badge {display:inline-block; padding:.12rem .45rem; border-radius:.4rem; 
        background:#1f2a37; color:#b8c2cc; font-size:.75rem; margin-left:.35rem;}
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

def _season_key(s: str) -> int:
    try:
        a, b = str(s).replace("-", "/").split("/")
        return int(a)*100 + int(b)
    except:
        return -1

def round_numeric_for_display(df: pd.DataFrame, ndigits: int = 3) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=["number"]).columns
    if len(num_cols):
        out[num_cols] = out[num_cols].round(ndigits)
    return out

# ---------- Nombres “deportivos” para métricas y campos ----------
METRIC_LABELS = {
    "Player": "Jugador", "Squad": "Equipo", "Season": "Temporada",
    "Rol_Tactico": "Rol táctico", "Comp": "Competición", "Min": "Minutos", "Age": "Edad",

    "Gls_per90": "Goles por 90 (Gls_per90)",
    "xG_per90": "Goles esperados por 90 (xG_per90)",
    "NPxG_per90": "Goles esperados sin penaltis por 90 (NPxG_per90)",
    "Sh_per90": "Tiros por 90 (Sh_per90)",
    "SoT_per90": "Tiros a puerta por 90 (SoT_per90)",
    "G/SoT_per90": "Goles por tiro a puerta por 90 (G/SoT_per90)",

    "xA_per90": "Asistencias esperadas por 90 (xA_per90)",
    "xAG_per90": "Asistencias + Goles esperados por 90 (xAG_per90)",
    "KP_per90": "Pases clave por 90 (KP_per90)",
    "GCA90_per90": "Acciones que generan gol por 90 (GCA90_per90)",
    "SCA_per90": "Acciones que generan tiro por 90 (SCA_per90)",
    "1/3_per90": "Recuperaciones en último tercio por 90 (1/3_per90)",
    "PPA_per90": "Pases al área penal por 90 (PPA_per90)",

    "PrgP_per90": "Pases progresivos por 90 (PrgP_per90)",
    "PrgC_per90": "Conducciones progresivas por 90 (PrgC_per90)",
    "Carries_per90": "Conducciones por 90 (Carries_per90)",
    "TotDist_per90": "Distancia total por 90 (TotDist_per90)",

    "Tkl+Int_per90": "Entradas + Intercepciones por 90 (Tkl+Int_per90)",
    "Int_per90": "Intercepciones por 90 (Int_per90)",
    "Recov_per90": "Recuperaciones por 90 (Recov_per90)",
    "Blocks_per90": "Bloqueos por 90 (Blocks_per90)",
    "Clr_per90": "Despejes por 90 (Clr_per90)",

    "Touches_per90": "Toques por 90 (Touches_per90)",
    "Dis_per90": "Pérdidas por 90 (Dis_per90)",
    "Pressures_per90": "Presiones por 90 (Pressures_per90)",
    "Err_per90": "Errores por 90 (Err_per90)",

    "Cmp%": "Porcentaje Precisión de pase (Cmp%)",
    "Cmp_per90": "Pases completados por 90 (Cmp_per90)",

    "Save%": "Porcentajes de Paradas (Save%)",
    "PSxG+/-_per90": "Goles evitados por 90 (PSxG+/-_per90)",
    "PSxG_per90": "Calidad de tiros recibidos por 90 (PSxG_per90)",
    "Saves_per90": "Paradas por 90 (Saves_per90)",
    "CS%": "Porcentaje de Porterías a cero (CS%)",
    "Launch%": "Porcentaje de Saques largos (Launch%)",
}

def label(col: str) -> str:
    return METRIC_LABELS.get(col, col)

def labels_for(cols) -> dict:
    return {c: label(c) for c in cols}

def rename_for_display(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    mapping = {c: label(c) for c in cols if c in df.columns}
    return df[cols].rename(columns=mapping)

# ===================== Query params (nueva API) ==========
params = dict(st.query_params)
def _to_list(v): return [] if v is None else (v if isinstance(v, list) else [v])

player_pre = _to_list(params.get("player"))
squad_pre  = _to_list(params.get("squad"))
comp_pre   = _to_list(params.get("comp"))
rol_pre    = _to_list(params.get("rol"))
season_pre = _to_list(params.get("season"))
min_pre    = int(params.get("min", 900))
age_from   = int(params.get("age_from", 15))
age_to     = int(params.get("age_to", 40))

# ===================== Filtros (orden solicitado) ====================
st.sidebar.header("Filtros")

# 1) Jugador (con buscador)
player_opts = sorted(df["Player"].dropna().unique())
player = st.sidebar.multiselect("Jugador", player_opts, default=player_pre)

# 2) Equipo (con buscador)
squad_opts = sorted(df["Squad"].dropna().unique())
squad = st.sidebar.multiselect("Equipo", squad_opts, default=squad_pre)

# 3) Competición (con buscador)
comp_opts = sorted(df["Comp"].dropna().unique())
comp = st.sidebar.multiselect("Competición", comp_opts, default=comp_pre)

# 4) Ámbito temporal / Temporada
season_opts_all = sorted(df["Season"].dropna().unique(), key=_season_key)
current_season = season_opts_all[-1] if season_opts_all else None

scope = st.sidebar.radio(
    "Ámbito temporal",
    options=["Histórico (≥900′)", "Temporada en curso"],
    index=0,
)

if scope == "Histórico (≥900′)":
    default_hist = [s for s in season_opts_all if s != current_season] or season_opts_all
    season = st.sidebar.multiselect("Temporada (histórico)", season_opts_all, default=default_hist)
else:
    season = [current_season] if current_season else []
    st.sidebar.write(f"**Temporada en curso:** {current_season or '—'}")

# 5) Rol táctico (posición)
rol_opts = sorted(df["Rol_Tactico"].dropna().unique())
rol = st.sidebar.multiselect("Rol táctico (posición)", rol_opts, default=rol_pre)

# 6) Edad (rango) — solo slider
age_num = pd.to_numeric(df.get("Age", pd.Series(dtype=float)), errors="coerce")
if age_num.size:
    age_min, age_max = int(np.nanmin(age_num)), int(np.nanmax(age_num))
else:
    age_min, age_max = 15, 40
age_default = (max(age_min, age_from), min(age_max, age_to))
age_range = st.sidebar.slider("Edad (rango)", min_value=age_min, max_value=age_max,
                              value=age_default, key="age_slider_only")

# 7) Minutos jugados (≥) — solo slider
if scope == "Histórico (≥900′)":
    global_min = max(900, int(df.get("Min", pd.Series([900])).min())) if "Min" in df else 900
    global_max = int(df.get("Min", pd.Series([3420])).max()) if "Min" in df else 3420
    default_min = int(np.clip(900, global_min, global_max))
    min_sel = st.sidebar.slider("Minutos jugados (≥)", min_value=global_min, max_value=global_max,
                                value=default_min, key="mins_slider_hist_only")
else:
    cur_df = df[df["Season"].isin(season)] if season else df
    cur_max = int(cur_df.get("Min", pd.Series([0])).max()) if not cur_df.empty else 0
    cur_default = min(90, cur_max) if cur_max else 0
    min_sel = st.sidebar.slider("Minutos jugados (≥)", min_value=0, max_value=cur_max,
                                value=int(cur_default), step=30, key="mins_slider_cur_only")
    if min_sel < 900:
        st.sidebar.caption("🔎 Estás viendo muestras <900′ (muestra parcial).")

# --------- Subconjunto activo ----------
mask_common = True
if player: mask_common &= df["Player"].isin(player)
if squad:  mask_common &= df["Squad"].isin(squad)
if comp:   mask_common &= df["Comp"].isin(comp)
if rol:    mask_common &= df["Rol_Tactico"].isin(rol)
if season: mask_common &= df["Season"].isin(season)

if "Age" in df:
    age_series = pd.to_numeric(df["Age"], errors="coerce")
    mask_common &= age_series.between(age_range[0], age_range[1])

mask_minutes = (df["Min"] >= min_sel) if "Min" in df else True
dff_view = df.loc[mask_common & mask_minutes].copy()

# Persistir en URL
st.query_params.update({
    "player": player, "squad": squad, "comp": comp, "rol": rol, "season": season,
    "min": str(min_sel),
    "age_from": str(age_range[0]), "age_to": str(age_range[1]),
})

# ===================== Métricas ==========================
out_metrics = [
    "Gls_per90","xG_per90","NPxG_per90","Sh_per90","SoT_per90","G/SoT_per90",
    "xA_per90","KP_per90","GCA90_per90","SCA_per90",
    "PrgP_per90","PrgC_per90","Carries_per90",
    "Cmp%","Cmp_per90","Tkl+Int_per90","Int_per90","Recov_per90"
]
metrics_all = [m for m in out_metrics if m in dff_view.columns]

# ===================== Tabs ==============================
tab_overview, tab_ranking, tab_compare, tab_similarity, tab_shortlist = st.tabs(
    ["📊 Overview", "🏆 Ranking", "🆚 Comparador", "🧬 Similares", "⭐ Shortlist"]
)

# ===================== Guardas ===========================
def stop_if_empty(dfx):
    if len(dfx) == 0:
        st.warning("No hay jugadores que cumplan con estas condiciones de filtro. "
                   "Prueba a reducir el umbral de minutos, ampliar edades o seleccionar más roles/temporadas.")
        st.stop()

# --------- Overview (FUNCIÓN) ----------
def render_overview_block(df_in: pd.DataFrame) -> None:
    k1, k2, k3, k4 = st.columns(4, gap="large")
    with k1:
        st.metric("Jugadores (en filtro)", f"{len(df_in):,}")
    with k2:
        st.metric("Equipos (en filtro)", f"{df_in['Squad'].nunique()}")
    with k3:
        try:
            st.metric("Media de edad (en filtro)", f"{pd.to_numeric(df_in['Age'], errors='coerce').mean():.1f}")
        except Exception:
            st.metric("Media de edad (en filtro)", "—")
    with k4:
        med = int(df_in["Min"].median()) if "Min" in df_in and len(df_in) else 0
        st.metric("Minutos medianos (en filtro)", f"{med:,}")

    st.markdown("### Productividad ofensiva: **xG/90 vs Goles/90**")
    if all(c in df_in.columns for c in ["xG_per90","Gls_per90"]):
        fig = px.scatter(
            df_in, x="xG_per90", y="Gls_per90",
            color="Rol_Tactico",
            size=df_in.get("SoT_per90", None),
            hover_name="Player",
            labels=labels_for(["xG_per90","Gls_per90","Rol_Tactico","SoT_per90"]),
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Creación: **xA/90 vs Pases clave/90** (tamaño = GCA/90)")
    if all(c in df_in.columns for c in ["xA_per90","KP_per90","GCA90_per90"]):
        fig = px.scatter(
            df_in, x="xA_per90", y="KP_per90",
            size="GCA90_per90",
            color="Rol_Tactico",
            hover_name="Player",
            labels=labels_for(["xA_per90","KP_per90","GCA90_per90","Rol_Tactico"]),
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Progresión: **Top 15** en Pases progresivos por 90")
    if "PrgP_per90" in df_in.columns:
        top_prog = df_in.sort_values("PrgP_per90", ascending=False).head(15)
        fig = px.bar(
            top_prog.sort_values("PrgP_per90"),
            x="PrgP_per90", y="Player",
            color="Rol_Tactico",
            labels=labels_for(["PrgP_per90","Player","Rol_Tactico"]),
            template="plotly_dark",
            orientation="h"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Defensa: **Tkl+Int/90 vs Recuperaciones/90** (tamaño = Intercepciones/90)")
    if all(c in df_in.columns for c in ["Tkl+Int_per90","Recov_per90","Int_per90"]):
        fig = px.scatter(
            df_in, x="Tkl+Int_per90", y="Recov_per90",
            size="Int_per90",
            color="Rol_Tactico",
            hover_name="Player",
            labels=labels_for(["Tkl+Int_per90","Recov_per90","Int_per90","Rol_Tactico"]),
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Pase: **Precisión** vs **Volumen**")
    if all(c in df_in.columns for c in ["Cmp%","Cmp_per90"]):
        fig = px.scatter(
            df_in, x="Cmp%", y="Cmp_per90",
            color="Rol_Tactico",
            hover_name="Player",
            labels=labels_for(["Cmp%","Cmp_per90","Rol_Tactico"]),
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    if "Save%" in df_in.columns and "PSxG+/-_per90" in df_in.columns and "Saves_per90" in df_in.columns:
        gk_df = df_in[df_in["Rol_Tactico"].str.contains("GK|Portero", case=False, na=False)].copy()
        if len(gk_df):
            st.markdown("### Porteros: **% Paradas** vs **PSxG+/- por 90** (tamaño = Paradas/90)")
            fig = px.scatter(
                gk_df, x="Save%", y="PSxG+/-_per90",
                size="Saves_per90",
                hover_name="Player",
                color="Rol_Tactico",
                labels=labels_for(["Save%","PSxG+/-_per90","Saves_per90","Rol_Tactico"]),
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

# ===================== OVERVIEW ==========================
with tab_overview:
    stop_if_empty(dff_view)
    render_overview_block(dff_view)

# ===================== RANKING ===========================
with tab_ranking:
    stop_if_empty(dff_view)
    st.subheader("Ranking por métrica")

    # ---------- CSS: modo compacto (menos interlineado) ----------
    st.markdown("""
    <style>
      .rank-compact .stMarkdown, .rank-compact .stCaption, .rank-compact p {margin:0 0 .15rem 0;}
      .rank-compact [data-baseweb="select"]{margin:0 0 .30rem 0;}
      .rank-compact [data-baseweb="radio"]{margin:0 0 .30rem 0;}
      .rank-compact [data-baseweb="slider"]{margin:.15rem 0 .30rem 0;}
      .rank-compact .stButton, .rank-compact .stDownloadButton {margin:.25rem 0 .20rem 0;}
      .rank-compact [data-testid="stHorizontalBlock"] {row-gap:.25rem;}
      .rank-compact [data-testid="metric-container"] {padding:0 0 .25rem 0;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="rank-compact">', unsafe_allow_html=True)

    # ---------- KPIs de contexto ----------
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Jugadores (filtro global)", f"{len(dff_view):,}")
    with k2: st.metric("Equipos (filtro global)", f"{dff_view['Squad'].nunique():,}")
    with k3:
        try: st.metric("Media edad", f"{pd.to_numeric(dff_view['Age'], errors='coerce').mean():.1f}")
        except Exception: st.metric("Media edad", "—")
    with k4:
        med = int(dff_view["Min"].median()) if "Min" in dff_view and len(dff_view) else 0
        st.metric("Minutos medianos", f"{med:,}")

    # ---------- Filtro rápido de edad (solo ranking) ----------
    st.caption("Filtro edad rápida")
    quick_age = st.radio("", ["Todos", "U22 (≤22)", "U28 (≤28)"], horizontal=True, key="quick_age_rank")

    df_base = dff_view.copy()
    if "Age" in df_base.columns and quick_age != "Todos":
        age_num = pd.to_numeric(df_base["Age"], errors="coerce")
        if quick_age.startswith("U22"):
            df_base = df_base[age_num.le(22)]
        elif quick_age.startswith("U28"):
            df_base = df_base[age_num.le(28)]
    stop_if_empty(df_base)

    # ---------- Métricas por rol ----------
    GK_METRICS = [
        "Save%","PSxG+/-_per90","PSxG_per90","Saves_per90","CS%","Launch%"
    ]
    FP_METRICS = [
        "Gls_per90","xG_per90","NPxG_per90","Sh_per90","SoT_per90","G/SoT_per90",
        "xA_per90","KP_per90","GCA90_per90","SCA_per90",
        "PrgP_per90","PrgC_per90","Carries_per90","TotDist_per90",
        "Tkl+Int_per90","Int_per90","Recov_per90","Blocks_per90","Clr_per90",
        "Touches_per90","Dis_per90","Pressures_per90","Err_per90",
        "Cmp%","Cmp_per90","1/3_per90","PPA_per90"
    ]

    # si el usuario filtró "Portero" en el sidebar, mostramos SOLO GK
    is_gk_view = False
    try:
        # variable 'rol' viene del sidebar (multiselect de Rol_Tactico)
        if rol and any(("portero" in str(r).lower()) or ("gk" in str(r).lower()) for r in rol):
            is_gk_view = True
    except NameError:
        is_gk_view = False

    metrics_pool = GK_METRICS if is_gk_view else FP_METRICS
    metrics_all = [m for m in metrics_pool if m in df_base.columns]

    # ---------- Modo de ranking ----------
    st.caption("Modo de ordenación")
    rank_mode = st.radio("", ["Por una métrica", "Multi-métrica (ponderado)"],
                         horizontal=True, key="rank_mode")

    # Utilidades
    LOWER_IS_BETTER = {"Err_per90", "Dis_per90"}

    def _pct_series(s: pd.Series, lower_better: bool) -> pd.Series:
        p = s.rank(pct=True)
        return (1 - p) if lower_better else p

    def _age_band(x):
        try:
            a = float(x)
            if a <= 22: return "U22"
            if a <= 28: return "U28"
        except Exception:
            pass
        return ""

    # ---------- POR UNA MÉTRICA ----------
    if rank_mode == "Por una métrica":
        st.caption("Métrica para ordenar")
        metric_to_rank = st.selectbox(
            "", options=metrics_all, index=0 if metrics_all else None,
            format_func=lambda c: label(c), key="rank_metric"
        )

        st.caption("Orden")
        order_dir = st.radio("", ["Descendente (mejor arriba)", "Ascendente (peor arriba)"],
                             horizontal=True, key="rank_order")
        ascending = order_dir.startswith("Asc")
        lower_better = metric_to_rank in LOWER_IS_BETTER

        # Construcción
        cols_id = ["Player","Squad","Season","Rol_Tactico","Comp","Min","Age"]
        cols_metrics = [m for m in metrics_all if m in df_base.columns]
        df_full = df_base[cols_id + cols_metrics].copy()
        df_full["Edad (U22/U28)"] = df_full["Age"].apply(_age_band)

        df_full["Rank"] = df_full[metric_to_rank].rank(
            ascending=lower_better, method="min"
        ).astype(int)
        df_full["Pct (muestra)"] = (_pct_series(df_full[metric_to_rank], lower_better)*100).round(1)
        if "Rol_Tactico" in df_full.columns:
            df_full["Pct (por rol)"] = df_full.groupby("Rol_Tactico")[metric_to_rank] \
                .transform(lambda s: (_pct_series(s, lower_better)*100)).round(1)
        if "Comp" in df_full.columns:
            df_full["Pct (por comp)"] = df_full.groupby("Comp")[metric_to_rank] \
                .transform(lambda s: (_pct_series(s, lower_better)*100)).round(1)

        df_full = df_full.sort_values(metric_to_rank, ascending=ascending)
        n_total_rows = len(df_full)

        # Top N
        cols_top = st.columns([0.25, 0.25, 0.5])
        with cols_top[0]:
            show_all = st.checkbox("Mostrar todos", value=False, key="rank_show_all")
        with cols_top[1]:
            topn = n_total_rows if show_all else st.slider("Top N", 5, max(50, min(1000, n_total_rows)),
                                                           min(100, n_total_rows), key="rank_topn")
        with cols_top[2]:
            st.caption(f"Mostrando **1–{min(topn, n_total_rows)}** de **{n_total_rows}**")

        df_view = df_full.head(topn).copy()
        cols_ctx = [c for c in ["Rank","Pct (muestra)","Pct (por rol)","Pct (por comp)"] if c in df_view.columns]
        cols_show = ["Player","Squad","Season","Rol_Tactico","Comp","Min","Age","Edad (U22/U28)"] + cols_ctx + cols_metrics

        tabla_disp_num = round_numeric_for_display(df_view, ndigits=3)
        tabla_disp = rename_for_display(tabla_disp_num, cols_show)

        st.caption("🔎 Percentiles calculados sobre la **muestra filtrada** (incluye filtro rápido U22/U28).")

    # ---------- MULTI-MÉTRICA ----------
    else:
        # Nota de índices (explicación) colocada arriba del bloque multi-métrica
        st.caption("Índices: **Índice ponderado (0–100)** con métricas normalizadas + **Índice Final Métrica** (suma ponderada cruda).")

        ROLE_PRESETS = {
            "Portero":   ["Save%", "PSxG+/-_per90", "PSxG_per90", "Saves_per90", "CS%"],
            "Central":   ["Tkl+Int_per90", "Int_per90", "Blocks_per90", "Clr_per90", "Recov_per90"],
            "Lateral":   ["PPA_per90", "PrgP_per90", "Carries_per90", "Tkl+Int_per90", "1/3_per90"],
            "Mediocentro": ["xA_per90", "PrgP_per90", "Recov_per90", "Pressures_per90", "TotDist_per90"],
            "Volante":  ["xA_per90", "KP_per90", "GCA90_per90", "PrgP_per90", "SCA_per90"],
            "Delantero":["Gls_per90", "xG_per90", "NPxG_per90", "SoT_per90", "xA_per90"],
        }

        # Selector de preset + botón aplicar (siempre 5 métricas)
        c1, c2 = st.columns([0.7, 0.3])
        with c1:
            preset_sel = st.selectbox("Preset por rol (opcional)",
                                      ["— (personalizado)"] + list(ROLE_PRESETS.keys()),
                                      index=0, key="mm_preset")
        with c2:
            if preset_sel != "— (personalizado)":
                if st.button("Aplicar preset", use_container_width=True):
                    preset_feats = [m for m in ROLE_PRESETS[preset_sel] if m in metrics_all][:5]
                    st.session_state["mm_feats"] = preset_feats
                    st.success(f"Preset aplicado: {preset_sel} → {len(preset_feats)} métricas.")

        # Selección de métricas (por pool acorde al rol)
        mm_feats = st.multiselect(
            "Elige 3–12 métricas para construir el índice",
            options=metrics_all,
            default=st.session_state.get("mm_feats", metrics_all[:5]),
            format_func=lambda c: label(c),
            key="mm_feats"
        )
        if len(mm_feats) < 3:
            st.info("Selecciona al menos 3 métricas.")
            st.stop()

        # Pesos 0.0–2.0
        weights = {}
        with st.expander("⚖️ Pesos por métrica (0.0–2.0)", expanded=True):
            for f in mm_feats:
                weights[f] = st.slider(label(f), 0.0, 2.0, 1.0, 0.1, key=f"rankw_{f}")

        # Datos y limpieza
        X = df_base[mm_feats].astype(float).copy()
        for c in mm_feats:
            X[c] = X[c].fillna(X[c].median())

        # Índices
        Xn = (X - X.min()) / (X.max() - X.min() + 1e-9)
        import numpy as _np
        w_vec = _np.array([weights[f] for f in mm_feats], dtype=float)
        w_norm = w_vec / (w_vec.sum() + 1e-9)
        idx_norm_0_100 = (Xn.values @ w_norm) * 100.0           # 0..100
        idx_final_metrica = (X.values @ w_vec)                  # crudo

        df_rank = df_base[["Player","Squad","Season","Rol_Tactico","Comp","Min","Age"]].copy()
        df_rank["Edad (U22/U28)"] = df_rank["Age"].apply(_age_band)
        for f in mm_feats:
            df_rank[f] = df_base[f].astype(float)
        df_rank["Índice ponderado"] = idx_norm_0_100
        df_rank["Índice Final Métrica"] = idx_final_metrica

        # Top N
        df_rank = df_rank.sort_values("Índice Final Métrica", ascending=False)
        n_total_rows = len(df_rank)
        cols_top = st.columns([0.25, 0.25, 0.5])
        with cols_top[0]:
            show_all = st.checkbox("Mostrar todos", value=False, key="rank_show_all")
        with cols_top[1]:
            topn = n_total_rows if show_all else st.slider("Top N", 5, max(50, min(1000, n_total_rows)),
                                                           min(50, n_total_rows), key="rank_topn_mm")
        with cols_top[2]:
            st.caption(f"Mostrando **1–{min(topn, n_total_rows)}** de **{n_total_rows}**")

        df_rank = df_rank.head(topn)
        tabla_disp_num = round_numeric_for_display(df_rank, ndigits=3)
        tabla_disp_num["Índice ponderado"] = pd.to_numeric(
            tabla_disp_num["Índice ponderado"], errors="coerce"
        ).round(1)

        cols_show = ["Player","Squad","Season","Rol_Tactico","Comp","Min","Age","Edad (U22/U28)",
                     "Índice ponderado", "Índice Final Métrica"] + mm_feats
        tabla_disp = rename_for_display(tabla_disp_num, cols_show)

    # ---------- Acciones justo encima de la tabla ----------
    actL, actR = st.columns([0.5, 0.5])
    with actL:
        clear_pressed = st.button("🧹 Eliminar filtros", use_container_width=True)
    with actR:
        st.download_button(
            "⬇️ Exportar ranking (CSV)",
            data=tabla_disp.to_csv(index=False).encode("utf-8-sig"),
            file_name="ranking_scouting.csv",
            mime="text/csv",
            use_container_width=True,
            key="rank_dl_top"
        )
    if clear_pressed:
        for k in [
            "Jugador","Equipo","Competición","Rol táctico (posición)","Edad (rango)","Ámbito temporal",
            "Temporada (histórico)","mins_slider_hist_only","mins_slider_cur_only","age_slider_only",
            "rank_mode","rank_metric","rank_order","rank_topn","rank_topn_mm","mm_feats","mm_preset",
            "quick_age_rank","rank_show_all"
        ]:
            st.session_state.pop(k, None)
        st.query_params.clear()
        st.rerun()

    # ---------- Tabla + Heatmap (verde→amarillo→rojo) ----------
    try:
        from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, ColumnsAutoSizeMode, JsCode

        gb = GridOptionsBuilder.from_dataframe(tabla_disp)
        gb.configure_default_column(sortable=True, filter=True, resizable=True, floatingFilter=True)

        # primeras columnas legibles
        gb.configure_column(label("Player"), pinned="left", minWidth=240, wrapText=True, autoHeight=True)
        gb.configure_column(label("Squad"),  minWidth=160, wrapText=True, autoHeight=True)
        gb.configure_column(label("Season"), minWidth=110)
        gb.configure_column(label("Rol_Tactico"), header_name=label("Rol_Tactico"), minWidth=150, wrapText=True, autoHeight=True)
        if "Edad (U22/U28)" in tabla_disp.columns:
            gb.configure_column("Edad (U22/U28)", minWidth=90)

        # Heatmap en percentiles / índice ponderado
        heat_cols = [c for c in tabla_disp.columns if c.startswith("Pct (")]
        if "Índice ponderado" in tabla_disp.columns:
            heat_cols.append("Índice ponderado")

        heat_js = JsCode("""
            function(params) {
                var v = Number(params.value);
                if (isNaN(v)) { return {}; }
                // clamp 0..100
                var p = Math.max(0, Math.min(100, v));
                // 0 -> rojo (0º), 50 -> amarillo (60º), 100 -> verde (120º)
                var hue = p * 1.2; 
                return {'backgroundColor': 'hsl(' + hue + ', 70%, 32%)', 'color': 'white'};
            }
        """)
        for c in heat_cols:
            gb.configure_column(c, cellStyle=heat_js)

        # zebra
        zebra_js = JsCode("""
            function(params) {
              if (params.node && params.node.rowIndex % 2 === 0) {
                return {'backgroundColor': 'rgba(255,255,255,0.03)'};
              }
              return {};
            }
        """)
        gb.configure_grid_options(getRowStyle=zebra_js, enableBrowserTooltips=True, rowHeight=36)

        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
        gb.configure_side_bar()
        grid_options = gb.build()

        AgGrid(
            tabla_disp,
            gridOptions=grid_options,
            theme="streamlit",
            update_mode=GridUpdateMode.NO_UPDATE,
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
            fit_columns_on_grid_load=False,
            height=580,
            allow_unsafe_jscode=True,
        )
    except Exception:
        st.dataframe(tabla_disp, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True) 
        
# ===================== COMPARADOR (Radar) ===========================
with tab_compare:
    stop_if_empty(dff_view)
    st.subheader("Comparador de jugadores (Radar)")

    # --- Estilos compactos (solo este bloque) ---
    st.markdown("""
    <style>
    .cmp .block-container, .cmp [data-testid="stVerticalBlock"]{gap:.35rem !important}
    .cmp .stRadio, .cmp .stMultiSelect, .cmp .stSelectbox, .cmp .stSlider,
    .cmp .stToggleSwitch{margin: .1rem 0 .35rem 0 !important}
    .cmp label{margin-bottom:.15rem !important}
    .cmp .metric-note{color:#9aa2ad; font-size:.85rem; margin:.2rem 0 .6rem 0}
    .cmp .stMetric{padding-top:.2rem}
    </style>
    """, unsafe_allow_html=True)

    c = st.container()
    c.markdown('<div class="cmp">', unsafe_allow_html=True)

    # ---------- PLACEHOLDER KPI (se pintará por encima de los filtros) ----------
    kpi_top = c.container()
    c.caption(
        '<div class="metric-note">Índice agregado (0–100): media de métricas normalizadas seleccionadas. '
        'El Δ indica cuánto está por encima/por debajo del jugador de referencia.</div>',
        unsafe_allow_html=True
    )

    # ---------- PRESETS POR ROL ----------
    ROLE_PRESETS = {
        "Portero":     ["Save%", "PSxG+/-_per90", "PSxG_per90", "Saves_per90", "CS%"],
        "Central":     ["Tkl+Int_per90", "Int_per90", "Blocks_per90", "Clr_per90", "Recov_per90"],
        "Lateral":     ["PPA_per90", "PrgP_per90", "Carries_per90", "Tkl+Int_per90", "1/3_per90"],
        "Mediocentro": ["xA_per90", "PrgP_per90", "Recov_per90", "Pressures_per90", "TotDist_per90"],
        "Volante":     ["xA_per90", "KP_per90", "GCA90_per90", "PrgP_per90", "SCA_per90"],
        "Delantero":   ["Gls_per90", "xG_per90", "NPxG_per90", "SoT_per90", "xA_per90"],
    }

    # ================== FILTROS (APARECEN DEBAJO DEL KPI) ==================
    players_all = dff_view["Player"].dropna().unique().tolist()
    pre_sel = st.session_state.get("cmp_players", [])
    default_players = [p for p in pre_sel if p in players_all][:3] or players_all[:2]

    sel_players = c.multiselect("Jugadores (máx. 3)", players_all, default=default_players, key="cmp_players")
    if not sel_players:
        st.info("Selecciona al menos 1 jugador.")
        st.stop()
    if len(sel_players) > 3:
        sel_players = sel_players[:3]

    ref_player = c.selectbox("Jugador referencia (para Δ y percentiles)", sel_players, index=0, key="cmp_ref")

    col_r1, col_r2 = c.columns([0.72, 0.28])
    with col_r1:
        cmp_role = st.selectbox("Rol táctico (preset opcional)", ["— (ninguno)"] + list(ROLE_PRESETS.keys()),
                                index=0, key="cmp_role")
    with col_r2:
        if cmp_role != "— (ninguno)":
            if st.button("Aplicar preset", use_container_width=True, key="cmp_role_btn"):
                preset_feats = [m for m in ROLE_PRESETS[cmp_role] if m in dff_view.columns]
                st.session_state["feats"] = preset_feats
                st.success(f"Preset aplicado: {cmp_role} → {len(preset_feats)} métricas.")

    default_feats = st.session_state.get("feats", [c for c in dff_view.columns if c.endswith("_per90")][:6])
    radar_feats = c.multiselect(
        "Métricas para el radar (elige 4–10)",
        options=[c for c in dff_view.columns if c.endswith("_per90") or c in ["Cmp%","Save%"]],
        default=default_feats,
        key="feats",
        format_func=lambda c: label(c),
    )
    if len(radar_feats) < 4:
        st.info("Selecciona al menos 4 métricas para el radar.")
        st.stop()

    col_ctx1, col_ctx2, col_ctx3 = c.columns([1,1,1.2])
    ctx_mode = col_ctx1.selectbox(
        "Cálculo de percentiles",
        options=["Muestra filtrada", "Por rol táctico", "Por competición"],
        index=0,
        key="cmp_ctx",
    )
    show_baseline = col_ctx2.toggle("Mostrar baseline del grupo", value=True, key="cmp_baseline")
    use_percentiles = col_ctx3.toggle("Tooltip con percentiles", value=True, key="cmp_pct_tooltip")

    # ---------- Contexto de cálculo ----------
    def _ctx_mask(df_in: pd.DataFrame) -> pd.Series:
        if ctx_mode == "Muestra filtrada":
            return pd.Series(True, index=df_in.index)
        if ctx_mode == "Por rol táctico" and "Rol_Tactico" in df_in:
            if any(dff_view["Player"] == ref_player):
                rol_ref = dff_view.loc[dff_view["Player"] == ref_player, "Rol_Tactico"].iloc[0]
                return (df_in["Rol_Tactico"] == rol_ref)
        if ctx_mode == "Por competición" and "Comp" in df_in:
            if any(dff_view["Player"] == ref_player):
                comp_ref = dff_view.loc[dff_view["Player"] == ref_player, "Comp"].iloc[0]
                return (df_in["Comp"] == comp_ref)
        return pd.Series(True, index=df_in.index)

    df_group = dff_view[_ctx_mask(dff_view)].copy()
    if df_group.empty:
        df_group = dff_view.copy()

    # ---------- Normalización / percentiles ----------
    S = df_group[radar_feats].astype(float).copy()
    S_norm = (S - S.min()) / (S.max() - S.min() + 1e-9)
    baseline = S_norm.mean(axis=0)
    pct = df_group[radar_feats].rank(pct=True) if use_percentiles else None

    # ================== KPI (pintado ARRIBA) ==================
    with kpi_top:
        cols_kpi = st.columns(len(sel_players))
        ref_val = S_norm[df_group["Player"] == ref_player][radar_feats].mean(axis=1).mean() * 100
        for i, pl in enumerate(sel_players):
            val = S_norm[df_group["Player"] == pl][radar_feats].mean(axis=1).mean() * 100
            delta = None if pl == ref_player else round(val - float(ref_val), 1)
            cols_kpi[i].metric(pl + (" (ref.)" if pl == ref_player else ""), f"{val:,.1f}",
                               delta=None if delta is None else (f"{delta:+.1f}"))

    # ================== RADAR ==================
    theta_labels = [label(f) for f in radar_feats]
    fig = go.Figure()
    palette = ["#4F8BF9", "#F95F53", "#2BB673"]

    for i, pl in enumerate(sel_players):
        r_vec = S_norm[df_group["Player"] == pl][radar_feats].mean().fillna(0).values
        pct_pl = None
        if pct is not None:
            pct_pl = pct[df_group["Player"] == pl][radar_feats].mean()

        fig.add_trace(go.Scatterpolar(
            r=r_vec,
            theta=theta_labels,
            fill="toself",
            name=pl + (" (ref.)" if pl == ref_player else ""),
            line=dict(color=palette[i % len(palette)], width=2),
            opacity=0.85 if pl == ref_player else 0.7,
            hovertemplate="<b>%{theta}</b><br>Índice 0–1: %{r:.3f}"
                          + ("<br>Percentil: %{customdata:.0%}" if pct_pl is not None else "")
                          + "<extra></extra>",
            customdata=(pct_pl.values if pct_pl is not None else None),
        ))

    if show_baseline:
        fig.add_trace(go.Scatterpolar(
            r=baseline[radar_feats].values,
            theta=theta_labels,
            name="Baseline grupo",
            line=dict(dash="dash", color="#B9BEC6"),
            fill=None,
            hovertemplate="<b>%{theta}</b><br>Baseline: %{r:.3f}<extra></extra>",
        ))

    fig.update_layout(
        template="plotly_dark",
        polar=dict(radialaxis=dict(visible=True, range=[0,1], gridcolor="#374151", linecolor="#4b5563")),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0),
        margin=dict(l=30, r=30, t=10, b=10)
    )
    c.plotly_chart(fig, use_container_width=True)

    # ================== TABLA ==================
    raw_group = dff_view[_ctx_mask(dff_view)].copy()
    rows = {}
    for pl in sel_players:
        rows[pl] = raw_group[raw_group["Player"] == pl][radar_feats].astype(float).mean()

    df_cmp = pd.DataFrame({"Métrica": [label(f) for f in radar_feats]})
    for pl, vals in rows.items():
        df_cmp[pl] = vals.values
    for pl in sel_players:
        if pl == ref_player:
            continue
        df_cmp[f"Δ ({pl} − {ref_player})"] = df_cmp[pl] - df_cmp[ref_player]

    if use_percentiles:
        pct_raw = raw_group[radar_feats].rank(pct=True)
        for pl in sel_players:
            pr = pct_raw[raw_group["Player"] == pl][radar_feats].mean(numeric_only=True) * 100
            df_cmp[f"% {pl}"] = pr.values

    for ccol in df_cmp.columns:
        if ccol != "Métrica":
            df_cmp[ccol] = pd.to_numeric(df_cmp[ccol], errors="coerce").round(3)

    first_delta = [c for c in df_cmp.columns if c.startswith("Δ (")]
    if first_delta:
        df_cmp = df_cmp.reindex(df_cmp[first_delta[0]].abs().sort_values(ascending=False).index)

    c.caption(
        '<div class="metric-note"><b>Cómo leer:</b> columnas con nombre de jugador = valor por 90’. '
        '<b>Δ</b> = diferencia vs referencia · <b>%</b> = percentil en el grupo elegido.</div>',
        unsafe_allow_html=True
    )
    c.dataframe(df_cmp, use_container_width=True, hide_index=True)

    # Export PNG del radar (opcional)
    try:
        png_bytes = fig.to_image(format="png", scale=2)
        c.download_button("🖼️ Descargar radar (PNG)", data=png_bytes,
                          file_name=f"radar_{'_vs_'.join(sel_players)}.png",
                          mime="image/png", key="cmp_png_dl")
    except Exception:
        c.caption('Para exportar PNG instala <code>kaleido</code> en <code>requirements.txt</code>.',
                  unsafe_allow_html=True)

    c.markdown('</div>', unsafe_allow_html=True)


# ===================== SIMILARES (rediseñado y compacto) =========================
with tab_similarity:
    stop_if_empty(dff_view)
    st.subheader("Jugadores similares (cosine similarity)")

    # ---- CSS: compacta espaciado de controles ----
    st.markdown("""
    <style>
      .block-container div[data-testid="stHorizontalBlock"] { row-gap: .25rem !important; }
      .stSelectbox, .stMultiSelect, .stRadio, .stSlider, .stCheckbox { margin: .15rem 0 !important; }
      .sim-kpi .stMetric { text-align:center; }
      .sim-badge {display:inline-block;padding:.1rem .4rem;border-radius:.4rem;background:#1f2a37;color:#cdd6e3;font-size:.75rem;margin-left:.35rem}
    </style>
    """, unsafe_allow_html=True)



    GK_METRICS = [
        "Save%","PSxG+/-_per90","PSxG_per90","Saves_per90","CS%","Launch%"
    ]
    FP_METRICS = [
        "Gls_per90","xG_per90","NPxG_per90","Sh_per90","SoT_per90","G/SoT_per90",
        "xA_per90","KP_per90","GCA90_per90","SCA_per90",
        "PrgP_per90","PrgC_per90","Carries_per90","TotDist_per90",
        "Tkl+Int_per90","Int_per90","Recov_per90","Blocks_per90","Clr_per90",
        "Touches_per90","Dis_per90","Pressures_per90","Err_per90",
        "Cmp%","Cmp_per90","1/3_per90","PPA_per90"
    ]









        

    # ---------- Presets por rol (reutiliza los 5 de Ranking) ----------
    ROLE_PRESETS = {
        "Portero":   ["Save%", "PSxG+/-_per90", "PSxG_per90", "Saves_per90", "CS%", "Cmp_per90", "Err_per90"],
        "Central":   ["Tkl+Int_per90", "Int_per90", "Blocks_per90", "Clr_per90", "Recov_per90", "Cmp_per90", "Err_per90"],
        "Lateral":   ["PPA_per90", "PrgP_per90", "Carries_per90", "Tkl+Int_per90", "1/3_per90", "Touches_per90", "xA_per90"],
        "Mediocentro": ["xA_per90", "PrgP_per90", "Recov_per90", "Pressures_per90", "TotDist_per90", "1/3_per90", "Touches_per90"],
        "Volante":  ["xA_per90", "KP_per90", "GCA90_per90", "PrgP_per90", "SCA_per90", "PrgC_per90","G/SoT_per90"],
        "Delantero":["Gls_per90", "xG_per90", "NPxG_per90", "SoT_per90", "xA_per90", "1/3_per90","PrgC_per90"],
    }

    # ---------- 1) Controles (compactos) ----------
    players_all = dff_view["Player"].dropna().unique().tolist()
    sel_players = st.multiselect("Jugadores (máx. 3)", players_all,
                                 default=players_all[:2] if len(players_all)>=2 else players_all[:1],
                                 key="sim_players")
    if not sel_players:
        st.info("Selecciona al menos 1 jugador.")
        st.stop()
    if len(sel_players) > 3:
        sel_players = sel_players[:3]

    ref_player = st.selectbox("Jugador objetivo (para similitud)", sel_players, index=0, key="sim_ref")

    c1, c2 = st.columns([0.70, 0.30])
    with c1:
        preset_sel = st.selectbox("Rol táctico (preset opcional)",
                                  ["— (ninguno)"] + list(ROLE_PRESETS.keys()),
                                  index=0, key="sim_preset")
    with c2:
        if st.button("Aplicar preset de métricas", use_container_width=True):
            preset_feats = [m for m in ROLE_PRESETS.get(preset_sel, []) if m in dff_view.columns]
            st.session_state["sim_feats"] = preset_feats or st.session_state.get("sim_feats", [])
            st.success(f"Preset aplicado: {preset_sel} → {len(preset_feats)} métricas.")
            st.rerun()  # evita warnings y asegura sincronía

    # Multiselect de métricas (toma el preset si existe)
    feats_default = st.session_state.get("sim_feats", ROLE_PRESETS.get(preset_sel, []))
    options_all = [c for c in dff_view.columns if c.endswith("_per90") or c in ["Cmp%","Save%"]]
    feats_sim = st.multiselect("Selecciona 6–12 métricas",
                               options=options_all,
                               default=(feats_default if feats_default else options_all[:8]),
                               key="sim_feats",
                               format_func=lambda c: label(c))
    if len(feats_sim) < 6:
        st.info("Selecciona al menos 6 métricas para una similitud estable.")
        st.stop()

    # Normalización (contexto) + filtros de fichabilidad
    ctx1, ctx2, ctx3 = st.columns([1,1,1])
    ctx_mode = ctx1.selectbox("Normalización de métricas",
                              ["Muestra filtrada", "Por rol táctico", "Por competición"], index=0, key="sim_ctx")
    excl_same_team = ctx2.toggle("Excluir mismo equipo", value=False, key="sim_excl_team")
    excl_same_comp = ctx3.toggle("Excluir misma competición", value=False, key="sim_excl_comp")

    f1, f2, f3, f4 = st.columns([1,1,1,1])
    min_minutes = f1.number_input("Min. minutos", min_value=0, value=0, step=90)
    band = f2.selectbox("Banda de edad", ["Todas","U22 (≤22)","U28 (≤28)"], index=0)
    topn = f3.slider("Top N resultados", 5, 100, 25)
    show_short_btn = f4.toggle("Botón: añadir a shortlist", value=True)

    # ---------- 2) Pool según contexto ----------
    def _ctx_mask(df_in: pd.DataFrame) -> pd.Series:
        if ctx_mode == "Muestra filtrada":
            return pd.Series(True, index=df_in.index)
        if ctx_mode == "Por rol táctico" and "Rol_Tactico" in df_in:
            rol_ref = dff_view.loc[dff_view["Player"] == ref_player, "Rol_Tactico"].iloc[0] \
                      if any(dff_view["Player"] == ref_player) else None
            return (df_in["Rol_Tactico"] == rol_ref) if rol_ref is not None else pd.Series(True, index=df_in.index)
        if ctx_mode == "Por competición" and "Comp" in df_in:
            comp_ref = dff_view.loc[dff_view["Player"] == ref_player, "Comp"].iloc[0] \
                      if any(dff_view["Player"] == ref_player) else None
            return (df_in["Comp"] == comp_ref) if comp_ref is not None else pd.Series(True, index=df_in.index)
        return pd.Series(True, index=df_in.index)

    pool = dff_view[_ctx_mask(dff_view)].copy()
    stop_if_empty(pool)

    # Filtros operativos
    if "Min" in pool:
        pool = pool[pool["Min"].fillna(0) >= min_minutes]
    if band != "Todas" and "Age" in pool:
        age_num = pd.to_numeric(pool["Age"], errors="coerce")
        if band.startswith("U22"):
            pool = pool[age_num.le(22)]
        elif band.startswith("U28"):
            pool = pool[age_num.le(28)]

    if excl_same_team and "Squad" in pool.columns and any(pool["Player"] == ref_player):
        ref_team = pool.loc[pool["Player"] == ref_player, "Squad"].iloc[0]
        pool = pool[pool["Squad"] != ref_team]
    if excl_same_comp and "Comp" in pool.columns and any(pool["Player"] == ref_player):
        ref_comp = pool.loc[pool["Player"] == ref_player, "Comp"].iloc[0]
        pool = pool[pool["Comp"] != ref_comp]

    stop_if_empty(pool)

    # ---------- 3) KPI header ----------
    hk1, hk2, hk3, hk4 = st.columns(4, gap="small")
    hk1.metric("Jugadores en pool", f"{len(pool):,}")
    hk2.metric("Candidatos únicos", f"{pool['Player'].nunique():,}")
    hk3.metric("Edad media (pool)", f"{pd.to_numeric(pool['Age'], errors='coerce').mean():.1f}" if "Age" in pool else "—")

    # ---------- 4) Similitud (cosine) + explicabilidad ----------
    feats_sim = [f for f in feats_sim if f in pool.columns]  # seguridad
    X_raw = pool[feats_sim].astype(float).copy()
    # min-max por columna
    X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min() + 1e-9)
    X = X.fillna(0.0)

    import numpy as _np

    # vector objetivo (media si varias filas)
    if any(pool["Player"] == ref_player):
        v = X[pool["Player"] == ref_player].mean(axis=0).to_numpy()
        pool_no_ref = pool[pool["Player"] != ref_player].copy()
        X_no_ref = X.loc[pool_no_ref.index].copy()
    else:
        # si no está en pool, usa dff_view normalizado al mismo rango
        base = dff_view.copy()
        ref_row = base[base["Player"] == ref_player]
        if ref_row.empty:
            st.warning("El jugador objetivo no está en la muestra; usando media 0 para referencia.")
            v = _np.zeros(len(feats_sim))
        else:
            rr = ref_row[feats_sim].astype(float)
            rr = (rr - X_raw.min()) / (X_raw.max() - X_raw.min() + 1e-9)
            v = rr.fillna(0.0).mean(axis=0).to_numpy()
        pool_no_ref = pool.copy()
        X_no_ref = X.copy()

    # cosine
    v_unit = v / (_np.linalg.norm(v) + 1e-12)
    U = X_no_ref.to_numpy()
    U_unit = U / (_np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    sim = (U_unit @ v_unit)

    # contribuciones (por-qué se parece): U_unit * v_unit por columna
    contrib = U_unit * v_unit  # shape (n, d)
    feat_labels = [label(f) for f in feats_sim]
    idx_top3 = _np.argsort(-contrib, axis=1)[:, :3]
    why3 = [", ".join([feat_labels[i] for i in idx_row]) for idx_row in idx_top3]

    # Varianza media para KPI
    hk4.metric("Varianza media (métricas)", f"{float(X.var().mean()):.3f}")

    # ---------- 5) Perfil del objetivo: fortalezas vs pool ----------
    st.caption("**Perfil del objetivo** — percentiles vs pool en las métricas seleccionadas.")
    ref_mask = (pool["Player"] == ref_player)
    if ref_mask.any():
        S = pool[feats_sim].astype(float)
        pcts = S.rank(pct=True)
        ref_pct = pcts[ref_mask].mean().sort_values(ascending=False)
        top_str = ref_pct.head(5)
        top_weak = ref_pct.tail(5)
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Fortalezas (Top-5)**")
            for k, v_ in top_str.items():
                st.write(f"• {label(k)} — {v_*100:.0f}º pct")
        with colB:
            st.markdown("**Áreas a mejorar (Bottom-5)**")
            for k, v_ in top_weak.items():
                st.write(f"• {label(k)} — {v_*100:.0f}º pct")
    else:
        st.info("El objetivo no está en el pool actual; no se muestra su perfil contextual.")

    # ---------- 6) Tabla de similares (heatmap + shortlist + export) ----------
    out_cols_id = ["Player","Squad","Season","Rol_Tactico","Comp","Min","Age"]
    out = pool_no_ref[out_cols_id].copy()
    out["similarity"] = sim
    out["Top similitudes (3)"] = why3
    out = out.sort_values("similarity", ascending=False).head(topn)

    try:
        from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode, GridUpdateMode, JsCode
        disp = round_numeric_for_display(out, ndigits=3)
        disp = rename_for_display(disp, out_cols_id + ["similarity","Top similitudes (3)"])

        gb = GridOptionsBuilder.from_dataframe(disp)
        gb.configure_default_column(sortable=True, filter=True, resizable=True, floatingFilter=True)
        gb.configure_column(label("Player"), pinned="left", minWidth=230, tooltipField=label("Player"))

        # Heatmap similarity verde->amarillo->rojo (alto=mejor)
        heat_js = JsCode("""
            function(params) {
                var v = Number(params.value);
                if (isNaN(v)) { return {}; }
                v = Math.max(0, Math.min(1.0, v));
                var hue = 120 * v; // 0->rojo, 120->verde
                return {'backgroundColor': 'hsl(' + hue + ', 65%, 30%)', 'color': 'white'};
            }
        """)
        gb.configure_column("similarity", cellStyle=heat_js, minWidth=120)

        grid = AgGrid(
            disp, gridOptions=gb.build(),
            update_mode=GridUpdateMode.NO_UPDATE,
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
            theme="streamlit", height=460, allow_unsafe_jscode=True
        )
    except Exception:
        st.dataframe(out, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Exportar similares (CSV)",
        data=out.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"similares_{ref_player}.csv",
        mime="text/csv",
        key="sim_export"
    )

# ===================== SHORTLIST =========================
with tab_shortlist:
    stop_if_empty(dff_view)
    st.subheader("Shortlist (lista de seguimiento)")
    if "shortlist" not in st.session_state: st.session_state.shortlist = []
    to_add = st.multiselect("Añadir jugadores", dff_view["Player"].dropna().unique().tolist())
    if st.button("➕ Agregar seleccionados"): st.session_state.shortlist = sorted(set(st.session_state.shortlist) | set(to_add))
    to_remove = st.multiselect("Eliminar de shortlist", st.session_state.shortlist)
    if st.button("🗑️ Eliminar seleccionados"): st.session_state.shortlist = [p for p in st.session_state.shortlist if p not in set(to_remove)]
    sh = dff_view[dff_view["Player"].isin(st.session_state.shortlist)]

    base_cols = ["Player","Squad","Season","Rol_Tactico","Comp","Min","Age"]
    sh_disp_num = round_numeric_for_display(sh[base_cols], ndigits=3)
    st.dataframe(rename_for_display(sh_disp_num, base_cols), use_container_width=True)

    st.download_button(
        "⬇️ Descargar shortlist (CSV)",
        data=sh_disp_num.to_csv(index=False).encode("utf-8-sig"),
        file_name="shortlist_scouting.csv",
        mime="text/csv",
    )















