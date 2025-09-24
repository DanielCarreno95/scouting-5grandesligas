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

# --------- Construcción del subconjunto activo (dff_view) ----------
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

# --------- Overview ----------
def render_overview_block(df_in):
    k1, k2, k3, k4 = st.columns(4, gap="large")
    with k1: st.metric("Jugadores (en filtro)", f"{len(df_in):,}")
    with k2: st.metric("Equipos (en filtro)", f"{df_in['Squad'].nunique()}")
    with k3:
        try: st.metric("Media de edad (en filtro)", f"{pd.to_numeric(df_in['Age'], errors='coerce').mean():.1f}")
        except Exception: st.metric("Media de edad (en filtro)", "—")
    with k4:
        med = int(df_in["Min"].median()) if "Min" in df_in and len(df_in) else 0
        st.metric("Minutos medianos (en filtro)", f"{med:,}")

    st.markdown("### Productividad ofensiva: **xG/90 vs Goles/90**")
    if all(c in df_in.columns for c in ["xG_per90","Gls_per90"]):
        fig = px.scatter(
            df_in, x="xG_per90", y="Gls_per90",
            color="Rol_Tactico", size=df_in.get("SoT_per90", None),
            hover_name="Player",
            labels=labels_for(["xG_per90","Gls_per90","Rol_Tactico","SoT_per90"]),
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Creación: **xA/90 vs Pases clave/90** (tamaño = GCA/90)")
    if all(c in df_in.columns for c in ["xA_per90","KP_per90","GCA90_per90"]):
        fig = px.scatter(
            df_in, x="xA_per90", y="KP_per90",
            size="GCA90_per90", color="Rol_Tactico", hover_name="Player",
            labels=labels_for(["xA_per90","KP_per90","GCA90_per90","Rol_Tactico"]),
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Progresión: **Top 15** en Pases progresivos por 90")
    if "PrgP_per90" in df_in.columns:
        top_prog = df_in.sort_values("PrgP_per90", ascending=False).head(15)
        fig = px.bar(
            top_prog.sort_values("PrgP_per90"),
            x="PrgP_per90", y="Player", color="Rol_Tactico",
            labels=labels_for(["PrgP_per90","Player","Rol_Tactico"]),
            template="plotly_dark", orientation="h"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Defensa: **Tkl+Int/90 vs Recuperaciones/90** (tamaño = Intercepciones/90)")
    if all(c in df_in.columns for c in ["Tkl+Int_per90","Recov_per90","Int_per90"]):
        fig = px.scatter(
            df_in, x="Tkl+Int_per90", y="Recov_per90", size="Int_per90",
            color="Rol_Tactico", hover_name="Player",
            labels=labels_for(["Tkl+Int_per90","Recov_per90","Int_per90","Rol_Tactico"]),
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Pase: **Precisión** vs **Volumen**")
    if all(c in df_in.columns for c in ["Cmp%","Cmp_per90"]):
        fig = px.scatter(
            df_in, x="Cmp%", y="Cmp_per90", color="Rol_Tactico", hover_name="Player",
            labels=labels_for(["Cmp%","Cmp_per90","Rol_Tactico"]),
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    if "Save%" in df_in.columns and "PSxG+/-_per90" in df_in.columns and "Saves_per90" in df_in.columns:
        gk_df = df_in[df_in["Rol_Tactico"].str.contains("GK|Portero", case=False, na=False)].copy()
        if len(gk_df):
            st.markdown("### Porteros: **% Paradas** vs **PSxG+/- por 90** (tamaño = Paradas/90)")
            fig = px.scatter(
                gk_df, x="Save%", y="PSxG+/-_per90", size="Saves_per90",
                hover_name="Player", color="Rol_Tactico",
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

    metric_to_rank = st.selectbox(
        "Métrica para ordenar",
        options=metrics_all,
        index=0 if metrics_all else None,
        format_func=lambda c: label(c),
        key="rank_metric",
    )
    topn = st.slider("Top N", 5, 100, 20, key="rank_topn")

    cols_show = ["Player", "Squad", "Season", "Rol_Tactico", "Comp", "Min", "Age"] + metrics_all
    tabla = dff_view[cols_show].sort_values(metric_to_rank, ascending=False).head(topn)

    # Redondeo a 3 decimales SOLO para mostrar y renombre para display
    tabla_disp_num = round_numeric_for_display(tabla, ndigits=3)
    tabla_disp = rename_for_display(tabla_disp_num, cols_show)

    try:
        from st_aggrid import (
            AgGrid,
            GridOptionsBuilder,
            GridUpdateMode,
            ColumnsAutoSizeMode,
        )
        gb = GridOptionsBuilder.from_dataframe(tabla_disp)

        # Default col
        gb.configure_default_column(sortable=True, filter=True, resizable=True, floatingFilter=True)

        # ---- Anchos fijos y pin de Jugador ----
        gb.configure_column(label("Player"), pinned="left", width=230)
        gb.configure_column(label("Squad"), width=180)
        gb.configure_column(label("Season"), width=120)
        gb.configure_column(label("Rol_Tactico"), header_name=label("Rol_Tactico"), width=170)
        gb.configure_column(label("Comp"), width=170)
        gb.configure_column(label("Min"), width=110)
        gb.configure_column(label("Age"), width=90)

        # Paginación + sidebar
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
        gb.configure_side_bar()

        # Evitar que el autosize nos aplaste los anchos definidos
        gb.configure_grid_options(domLayout="normal")

        grid_options = gb.build()

        AgGrid(
            tabla_disp,
            gridOptions=grid_options,
            theme="streamlit",
            update_mode=GridUpdateMode.NO_UPDATE,
            columns_auto_size_mode=ColumnsAutoSizeMode.NO_AUTOSIZE,  # <- no tocar los anchos definidos
            fit_columns_on_grid_load=False,
            height=580,
            allow_unsafe_jscode=False,
        )
    except Exception:
        st.info("Para fijar columnas y anchos usa `streamlit-aggrid` en `requirements.txt`. Mostrando tabla estándar.")
        st.dataframe(tabla_disp, use_container_width=True)

# ===================== COMPARADOR ========================================
with tab_compare:
    # --- seguridad: si el df filtrado está vacío, aviso y paro esta pestaña ---
    if len(dff) == 0:
        st.warning("No hay jugadores con los filtros actuales. Ajusta los filtros para comparar.")
        st.stop()

    st.subheader("Comparador de jugadores (Radar)")

    # ---------- plantillas ----------
    TEMPLATES = {
        "⚽ Finalizador": ["Gls_per90", "xG_per90", "SoT_per90", "G/SoT_per90", "xA_per90", "KP_per90"],
        "🎯 Creador": ["xA_per90", "KP_per90", "GCA90_per90", "SCA_per90", "PrgP_per90", "Cmp%"],
        "🔋 Box-to-box": ["PrgC_per90", "Carries_per90", "Tkl+Int_per90", "Recov_per90", "Cmp_per90", "Gls_per90"],
        "🛡️ Central": ["Tkl+Int_per90", "Int_per90", "Recov_per90", "Clr_per90", "Cmp%", "PrgP_per90"],
        "🚀 Lateral": ["PrgC_per90", "PrgP_per90", "PPA_per90", "SCA_per90", "Tkl+Int_per90", "Recov_per90"],
        "🧤 Portero": ["Save%", "PSxG+/-_per90", "Saves_per90", "Launch%", "PSxG_per90", "CS%"],
    }
    # deja solo las métricas que existan en el dataset actual
    TEMPLATES = {k: [m for m in v if m in metrics_all] for k, v in TEMPLATES.items()}

    # ---------- selección de jugadores: hasta 3 ----------
    players_all = dff["Player"].dropna().unique().tolist()
    sel_players = st.multiselect(
        "Jugadores (máx. 3)",
        players_all,
        default=players_all[:2] if len(players_all) >= 2 else players_all[:1],
        key="cmp_players"
    )
    if len(sel_players) == 0:
        st.info("Selecciona al menos 1 jugador.")
        st.stop()
    if len(sel_players) > 3:
        st.info("Has seleccionado más de 3 jugadores. Se tomarán los 3 primeros.")
        sel_players = sel_players[:3]

    # ---------- contexto para percentiles/baseline ----------
    st.markdown("**Contexto de comparación**")
    col_ctx1, col_ctx2, col_ctx3 = st.columns([1,1,1.5])
    ctx_mode = col_ctx1.selectbox(
        "Cálculo de percentiles",
        options=["Muestra filtrada", "Por rol táctico", "Por competición"],
        index=0,
        help="Define contra qué grupo se calculan percentiles y baseline.",
        key="cmp_ctx",
    )
    show_baseline = col_ctx2.toggle("Mostrar baseline del grupo", value=True, key="cmp_baseline")
    use_percentiles = col_ctx3.toggle("Tooltip con percentiles", value=True, key="cmp_pct_tooltip")

    # ---------- plantillas y selección de métricas ----------
    col_tpl, col_ms = st.columns([1,3])
    tpl_name = col_tpl.selectbox("Plantillas de métricas", options=list(TEMPLATES.keys())+["(Personalizado)"], index=0, key="tpl")
    default_feats = TEMPLATES.get(tpl_name, metrics_all[:6])

    radar_feats = col_ms.multiselect(
        "Métricas para el radar (elige 4–10)",
        options=metrics_all,
        default=default_feats[: min(10, len(default_feats))] if default_feats else metrics_all[:6],
        key="feats",
        format_func=lambda c: label(c)
    )
    if len(radar_feats) < 4:
        st.info("Selecciona al menos 4 métricas para el radar.")
        st.stop()

    # ---------- pesos ----------
    weights = {f: 1.0 for f in radar_feats}
    with st.expander("⚖️ Ajustar pesos por métrica (0.5 — 2.0)", expanded=False):
        for f in radar_feats:
            weights[f] = st.slider(label(f), 0.5, 2.0, 1.0, 0.1, key=f"w_{f}")

    # ---------- helpers ----------
    def _normalize_0_1(df_num: pd.DataFrame) -> pd.DataFrame:
        mn = df_num.min(axis=0); mx = df_num.max(axis=0)
        return (df_num - mn) / (mx - mn + 1e-9)

    def _percentiles(df_grp: pd.DataFrame, feats: list) -> pd.DataFrame:
        return df_grp[feats].rank(pct=True)

    def _ctx_mask(df_in: pd.DataFrame) -> pd.Series:
        if ctx_mode == "Muestra filtrada":
            return pd.Series(True, index=df_in.index)
        # Referencia: primer jugador seleccionado
        ref = sel_players[0]
        if ctx_mode == "Por rol táctico" and "Rol_Tactico" in df_in:
            rol_ref = dff.loc[dff["Player"] == ref, "Rol_Tactico"].iloc[0] if any(dff["Player"] == ref) else None
            return (df_in["Rol_Tactico"] == rol_ref) if rol_ref is not None else pd.Series(True, index=df_in.index)
        if ctx_mode == "Por competición" and "Comp" in df_in:
            comp_ref = dff.loc[dff["Player"] == ref, "Comp"].iloc[0] if any(dff["Player"] == ref) else None
            return (df_in["Comp"] == comp_ref) if comp_ref is not None else pd.Series(True, index=df_in.index)
        return pd.Series(True, index=df_in.index)

    # ---------- grupo de contexto ----------
    df_group = dff[_ctx_mask(dff)].copy()
    if df_group.empty:
        st.warning("No hay jugadores en el grupo de contexto con los filtros actuales.")
        df_group = dff.copy()

    # normalización (aplicando pesos)
    S = df_group[radar_feats].astype(float).copy()
    for f in radar_feats:
        S[f] = S[f] * weights[f]
    S_norm = _normalize_0_1(S)
    baseline = S_norm.mean(axis=0)

    # percentiles (sobre crudo)
    pct = _percentiles(df_group, radar_feats) if use_percentiles else None

    # ---------- RADAR ----------
    theta_labels = [label(f) for f in radar_feats]
    fig = go.Figure()

    # paleta de colores para hasta 3 jugadores
    palette = [
        "#4F8BF9",  # azul
        "#F95F53",  # coral
        "#2BB673",  # verde
    ]

    for i, pl in enumerate(sel_players):
        # promedio por si hay varias filas del jugador
        vec = S_norm[df_group["Player"] == pl].mean(numeric_only=True).fillna(0)
        # percentiles para tooltip
        pct_pl = None
        if pct is not None:
            pct_pl = pct[df_group["Player"] == pl].mean(numeric_only=True)

        fig.add_trace(go.Scatterpolar(
            r=vec[radar_feats].values,
            theta=theta_labels,
            fill="toself",
            name=pl,
            line=dict(color=palette[i % len(palette)]),
            hovertemplate="<b>%{theta}</b><br>Index 0–1: %{r:.3f}"
                          + ("<br>Percentil: %{customdata:.0%}" if (pct_pl is not None) else "")
                          + "<extra></extra>",
            customdata=(pct_pl[radar_feats].values if pct_pl is not None else None),
        ))

    if show_baseline:
        fig.add_trace(go.Scatterpolar(
            r=baseline[radar_feats].values,
            theta=theta_labels,
            name="Baseline grupo",
            line=dict(dash="dash", color="#BBBBBB"),
            fill=None,
            hovertemplate="<b>%{theta}</b><br>Baseline: %{r:.3f}<extra></extra>"
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=True, margin=dict(l=30, r=30, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------- TABLA COMPARATIVA (crudo) ----------
    raw_group = dff[_ctx_mask(dff)].copy()
    rows = []
    for pl in sel_players:
        vals = raw_group[raw_group["Player"] == pl][radar_feats].astype(float).mean(numeric_only=True)
        rows.append(vals)

    df_cmp = pd.DataFrame({"Métrica": [label(f) for f in radar_feats]})
    for pl, vals in zip(sel_players, rows):
        df_cmp[pl] = vals.values

    # Δ (A - B) si hay al menos 2 jugadores
    if len(sel_players) >= 2:
        df_cmp[f"Δ ({sel_players[0]} - {sel_players[1]})"] = (rows[0] - rows[1]).values

    # percentiles crudos por jugador (opcional)
    if use_percentiles:
        pct_raw = _percentiles(raw_group, radar_feats)
        for pl in sel_players:
            pr = pct_raw[raw_group["Player"] == pl].mean(numeric_only=True)
            df_cmp[f"% {pl}"] = (pr.values * 100)

    # redondeo 3 decimales y orden por gap si existe Δ
    num_cols = [c for c in df_cmp.columns if c != "Métrica"]
    df_cmp[num_cols] = df_cmp[num_cols].astype(float).round(3)
    delta_cols = [c for c in df_cmp.columns if c.startswith("Δ (")]
    if delta_cols:
        df_cmp = df_cmp.reindex(df_cmp[delta_cols[0]].abs().sort_values(ascending=False).index)

    st.markdown("**Diferencias por métrica**")
    st.dataframe(df_cmp, use_container_width=True)

    # ---------- export ----------
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Descargar comparación (CSV)",
            data=df_cmp.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"comparacion_{'_vs_'.join(sel_players)}.csv",
            mime="text/csv",
            key="cmp_csv_dl"
        )
    with c2:
        try:
            import io
            png_bytes = fig.to_image(format="png", scale=2)  # requiere 'kaleido' en requirements.txt
            st.download_button(
                "🖼️ Descargar radar (PNG)",
                data=png_bytes,
                file_name=f"radar_{'_vs_'.join(sel_players)}.png",
                mime="image/png",
                key="cmp_png_dl"
            )
        except Exception:
            st.caption("ℹ️ Para exportar PNG instala **kaleido** en `requirements.txt`.")



# ===================== SIMILARES =========================
with tab_similarity:
    stop_if_empty(dff_view)
    st.subheader("Jugadores similares (cosine similarity)")
    feats_sim = st.multiselect(
        "Selecciona 6–12 métricas",
        options=metrics_all,
        default=metrics_all[:8],
        key="sim_feats",
        format_func=lambda c: label(c)
    )
    target = st.selectbox("Jugador objetivo", dff_view["Player"].dropna().unique().tolist())
    if feats_sim and target:
        X = dff_view[feats_sim].astype(float).fillna(0.0).to_numpy()
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-9)
        from numpy.linalg import norm
        idx = dff_view.index[dff_view["Player"]==target][0]
        v = X[dff_view.index.get_loc(idx)]
        sims = (X @ v) / (norm(X, axis=1)*norm(v) + 1e-9)

        out_cols = ["Player","Squad","Season","Rol_Tactico","Comp","Min","Age","similarity"]
        out = dff_view[["Player","Squad","Season","Rol_Tactico","Comp","Min","Age"]].copy()
        out["similarity"] = sims
        out = out.sort_values("similarity", ascending=False).head(25)

        out_disp_num = round_numeric_for_display(out, ndigits=3)
        st.dataframe(rename_for_display(out_disp_num, out_cols), use_container_width=True)

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

# ===================== Footer trazabilidad ===============
meta = next(DATA_DIR.glob("metadata_*.json"), None)
if meta and meta.exists():
    import json
    m = json.loads(meta.read_text(encoding="utf-8"))
    st.caption(f"📦 Dataset: {m.get('files',{}).get('parquet','parquet')} · "
               f"Filtros base: ≥{m.get('filters',{}).get('minutes_min',900)}′ · "
               f"Generado: {m.get('created_at','')}")


