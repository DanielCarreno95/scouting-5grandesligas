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

# ===================== OVERVIEW ==========================
with tab_overview:
    stop_if_empty(dff_view)
    render_overview_block(dff_view)

# ===================== RANKING ===========================
with tab_ranking:
    stop_if_empty(dff_view)
    st.subheader("Ranking por métrica")

    # ---------- CONTEXTO RÁPIDO (KPI banner) ----------
    df_kpi = dff_view.copy()
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Jugadores (filtro global)", f"{len(df_kpi):,}")
    with k2: st.metric("Equipos (filtro global)", f"{df_kpi['Squad'].nunique():,}")
    with k3:
        try: st.metric("Media edad", f"{pd.to_numeric(df_kpi['Age'], errors='coerce').mean():.1f}")
        except Exception: st.metric("Media edad", "—")
    with k4:
        med = int(df_kpi["Min"].median()) if "Min" in df_kpi and len(df_kpi) else 0
        st.metric("Minutos medianos", f"{med:,}")

    st.markdown("<hr style='opacity:.15;'>", unsafe_allow_html=True)

    # ---------- BLOQUE 1 · Filtro rápido de edad ----------
    st.caption("Filtro edad rápida")
    quick_age = st.radio(
        "", ["Todos", "U22 (≤22)", "U28 (≤28)"],
        horizontal=True, key="quick_age_rank"
    )

    # Base del ranking (aplica filtro rápido de edad sólo aquí)
    df_base = dff_view.copy()
    if "Age" in df_base.columns and quick_age != "Todos":
        age_num = pd.to_numeric(df_base["Age"], errors="coerce")
        if quick_age.startswith("U22"):
            df_base = df_base[age_num.le(22)]
        elif quick_age.startswith("U28"):
            df_base = df_base[age_num.le(28)]
    stop_if_empty(df_base)

    st.markdown("<hr style='opacity:.15;margin-top:.5rem;margin-bottom:.5rem;'>", unsafe_allow_html=True)

    # ---------- BLOQUE 2 · Modo de ranking ----------
    st.caption("Modo de ordenación")
    rank_mode = st.radio(
        "", ["Por una métrica", "Multi-métrica (ponderado)"],
        horizontal=True, key="rank_mode"
    )

    # Métricas disponibles con la muestra actual
    out_metrics = [
        "Gls_per90","xG_per90","NPxG_per90","Sh_per90","SoT_per90","G/SoT_per90",
        "xA_per90","KP_per90","GCA90_per90","SCA_per90","1/3_per90","PPA_per90",
        "PrgP_per90","PrgC_per90","Carries_per90","TotDist_per90",
        "Tkl+Int_per90","Int_per90","Recov_per90","Blocks_per90","Clr_per90",
        "Touches_per90","Dis_per90","Pressures_per90","Err_per90",
        "Cmp%","Cmp_per90",
        "Save%","PSxG+/-_per90","PSxG_per90","Saves_per90","CS%","Launch%",
    ]
    metrics_all = [m for m in out_metrics if m in df_base.columns]

    # Presets por rol (5 métricas clave)
    ROLE_PRESETS = {
        "Portero":   ["Save%", "PSxG+/-_per90", "PSxG_per90", "Saves_per90", "CS%"],
        "Central":   ["Tkl+Int_per90", "Int_per90", "Blocks_per90", "Clr_per90", "Recov_per90"],
        "Lateral":   ["PPA_per90", "PrgP_per90", "Carries_per90", "Tkl+Int_per90", "1/3_per90"],
        "Mediocentro": ["xA_per90", "PrgP_per90", "Recov_per90", "Pressures_per90", "TotDist_per90"],
        "Volante":  ["xA_per90", "KP_per90", "GCA90_per90", "PrgP_per90", "SCA_per90"],
        "Delantero":["Gls_per90", "xG_per90", "NPxG_per90", "SoT_per90", "xA_per90"],
    }

    def _age_band(x):
        try:
            a = float(x)
            if a <= 22: return "U22"
            if a <= 28: return "U28"
        except Exception:
            pass
        return ""

    # ---------- BLOQUE 3 · Parámetros según modo ----------
    LOWER_IS_BETTER = {"Err_per90", "Dis_per90"}

    def _pct_series(s: pd.Series, lower_better: bool) -> pd.Series:
        p = s.rank(pct=True)
        return (1 - p) if lower_better else p

    # Variables a completar
    tabla_disp = None
    n_total_rows = 0

    # ============== MODO: POR UNA MÉTRICA ==============
    if rank_mode == "Por una métrica":
        st.caption("Métrica para ordenar")
        metric_to_rank = st.selectbox(
            "", options=metrics_all,
            index=0 if metrics_all else None,
            format_func=lambda c: label(c), key="rank_metric",
        )

        st.caption("Orden")
        order_dir = st.radio(
            "", ["Descendente (mejor arriba)", "Ascendente (peor arriba)"],
            horizontal=True, key="rank_order"
        )
        ascending = order_dir.startswith("Asc")
        lower_better = metric_to_rank in LOWER_IS_BETTER

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

        # ---------- BLOQUE 4 · Top N + Mostrar todos ----------
        col_topL, col_topR = st.columns([0.65, 0.35])
        with col_topL:
            show_all = st.checkbox("Mostrar todos", value=False, key="rank_show_all")
            topn = n_total_rows if show_all else st.slider(
                "Top N", 5, max(50, min(1000, n_total_rows)), min(100, n_total_rows), key="rank_topn"
            )
            st.caption(f"Mostrando **1–{min(topn, n_total_rows)}** de **{n_total_rows}**")
        with col_topR:
            st.empty()

        df_view = df_full.head(topn).copy()
        cols_ctx = [c for c in ["Rank","Pct (muestra)","Pct (por rol)","Pct (por comp)"] if c in df_view.columns]
        cols_show = ["Player","Squad","Season","Rol_Tactico","Comp","Min","Age","Edad (U22/U28)"] + cols_ctx + cols_metrics

        tabla_disp_num = round_numeric_for_display(df_view, ndigits=3)
        tabla_disp = rename_for_display(tabla_disp_num, cols_show)

        st.caption("📌 Percentiles calculados sobre la **muestra filtrada** (incluye filtro rápido U22/U28).")

    # ============== MODO: MULTI-MÉTRICA ==============
    else:
        st.caption(
            '<div class="note">Índices: <b>Índice ponderado (0–100)</b> con métricas normalizadas y '
            '<b>Índice Final Métrica</b> (suma ponderada cruda).</div>',
            unsafe_allow_html=True
        )

        # Preset por rol
        col_p1, col_p2 = st.columns([0.7, 0.3])
        with col_p1:
            preset_sel = st.selectbox(
                "Preset por rol (opcional)", ["— (personalizado)"] + list(ROLE_PRESETS.keys()),
                index=0, key="mm_preset"
            )
        with col_p2:
            if preset_sel != "— (personalizado)":
                if st.button("Aplicar preset", use_container_width=True):
                    preset_feats = [m for m in ROLE_PRESETS[preset_sel] if m in metrics_all]
                    st.session_state["mm_feats"] = preset_feats
                    st.success(f"Preset aplicado: {preset_sel} → {len(preset_feats)} métricas.")

        # Selección de métricas
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

        # Índice 0–100
        Xn = (X - X.min()) / (X.max() - X.min() + 1e-9)
        import numpy as _np
        w_vec = _np.array([weights[f] for f in mm_feats], dtype=float)
        w_norm = w_vec / (w_vec.sum() + 1e-9)
        idx_norm_0_100 = (Xn.values @ w_norm) * 100.0

        # Índice Final Métrica (crudo)
        idx_final_metrica = (X.values @ w_vec)

        df_rank = df_base[["Player","Squad","Season","Rol_Tactico","Comp","Min","Age"]].copy()
        df_rank["Edad (U22/U28)"] = df_rank["Age"].app



