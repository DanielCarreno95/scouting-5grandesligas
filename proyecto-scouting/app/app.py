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

/* ==== Estilo del sidebar ==== */
[data-testid="stSidebar"] .sidebar-title { 
  font-size: 1.05rem; font-weight: 800; margin: .25rem 0 .5rem 0;
}
[data-testid="stSidebar"] label { 
  font-size: .98rem; font-weight: 600;
}
[data-testid="stSidebar"] .stMultiSelect, 
[data-testid="stSidebar"] .stSlider, 
[data-testid="stSidebar"] .stNumberInput, 
[data-testid="stSidebar"] .stRadio {
  margin-bottom: .6rem;
}
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

# ---------- Nombres “deportivos” ----------
METRIC_LABELS = {
    # Identidad / contexto
    "Player": "Jugador", "Squad": "Equipo", "Season": "Temporada",
    "Rol_Tactico": "Rol táctico", "Comp": "Competición", "Min": "Minutos", "Age": "Edad",

    # Ofensivo / finalización
    "Gls_per90": "Goles por 90 (Gls_per90)",
    "xG_per90": "Goles esperados por 90 (xG_per90)",
    "NPxG_per90": "Goles esperados sin penaltis por 90 (NPxG_per90)",
    "Sh_per90": "Tiros por 90 (Sh_per90)",
    "SoT_per90": "Tiros a puerta por 90 (SoT_per90)",
    "G/SoT_per90": "Goles por tiro a puerta por 90 (G/SoT_per90)",

    # Creatividad
    "xA_per90": "Asistencias esperadas por 90 (xA_per90)",
    "xAG_per90": "Asistencias + Goles esperados por 90 (xAG_per90)",
    "KP_per90": "Pases clave por 90 (KP_per90)",
    "GCA90_per90": "Acciones que generan gol por 90 (GCA90_per90)",
    "SCA_per90": "Acciones que generan tiro por 90 (SCA_per90)",
    "1/3_per90": "Recuperaciones en último tercio por 90 (1/3_per90)",
    "PPA_per90": "Pases al área penal por 90 (PPA_per90)",

    # Progresión
    "PrgP_per90": "Pases progresivos por 90 (PrgP_per90)",
    "PrgC_per90": "Conducciones progresivas por 90 (PrgC_per90)",
    "Carries_per90": "Conducciones por 90 (Carries_per90)",
    "TotDist_per90": "Distancia total por 90 (TotDist_per90)",

    # Defensa
    "Tkl+Int_per90": "Entradas + Intercepciones por 90 (Tkl+Int_per90)",
    "Int_per90": "Intercepciones por 90 (Int_per90)",
    "Recov_per90": "Recuperaciones por 90 (Recov_per90)",
    "Blocks_per90": "Bloqueos por 90 (Blocks_per90)",
    "Clr_per90": "Despejes por 90 (Clr_per90)",

    # Posesión / pérdidas / presión
    "Touches_per90": "Toques por 90 (Touches_per90)",
    "Dis_per90": "Pérdidas por 90 (Dis_per90)",
    "Pressures_per90": "Presiones por 90 (Pressures_per90)",
    "Err_per90": "Errores por 90 (Err_per90)",

    # Pase
    "Cmp%": "Porcentaje Precisión de pase (Cmp%)",
    "Cmp_per90": "Pases completados por 90 (Cmp_per90)",

    # Porteros
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

# ===================== Query params ======================
params = dict(st.query_params)
def _to_list(v): return [] if v is None else (v if isinstance(v, list) else [v])

comp_pre   = _to_list(params.get("comp"))
rol_pre    = _to_list(params.get("rol"))
season_pre = _to_list(params.get("season"))
min_pre    = int(params.get("min", 900))
age_from   = int(params.get("age_from", 15))
age_to     = int(params.get("age_to", 40))

# ===================== Filtros ===========================
st.sidebar.header("Filtros")

# Ajuste visual de tipografía en la barra lateral
st.markdown("""
<style>
div[data-testid="stSidebar"] label {font-size:.94rem; font-weight:600;}
div[data-testid="stSidebar"] .stMultiSelect, 
div[data-testid="stSidebar"] .stSelectbox, 
div[data-testid="stSidebar"] .stNumberInput, 
div[data-testid="stSidebar"] .stSlider {font-size:.92rem;}
</style>
""", unsafe_allow_html=True)

# --- Opciones base (orden alfabético) ---
player_all = sorted(df["Player"].dropna().unique())
squad_all  = sorted(df["Squad"].dropna().unique())
comp_all   = sorted(df["Comp"].dropna().unique())
rol_opts   = sorted(df["Rol_Tactico"].dropna().unique())

# Temporadas ordenadas por _season_key (ya definido arriba)
season_opts = sorted(df["Season"].dropna().unique(), key=_season_key)
current_season = season_opts[-1] if season_opts else None

# ---------- 1) Jugador (con buscador dentro del desplegable)
players_sel = multi_with_search("Jugador", player_all, key="filter_players")

# ---------- 2) Equipo (con buscador dentro del desplegable)
squads_sel = multi_with_search("Equipo", squad_all, key="filter_squads")

# ---------- 3) Competición (con buscador dentro del desplegable)
comp = multi_with_search(
    "Competición",
    comp_all,
    default=[c for c in comp_pre if c in comp_all] if 'comp_pre' in locals() else None,
    key="filter_comp"
)

# ---------- 4) Temporada (multiselección; el histórico/actual lo resuelves en Overview)
season = st.sidebar.multiselect(
    "Temporada",
    season_opts,
    default=season_pre if season_pre else season_opts,  # si no llega query param, todas
    key="filter_season"
)

# ---------- 5) Rol táctico (posición funcional)
rol = st.sidebar.multiselect(
    "Rol táctico (posición)",
    rol_opts,
    default=rol_pre if rol_pre else [],
    key="filter_rol"
)

# ---------- 6) Edad (rango) + inputs mínimos/máximos para precisión
age_num = pd.to_numeric(df.get("Age", pd.Series(dtype=float)), errors="coerce")
if age_num.size:
    age_min, age_max = int(np.nanmin(age_num)), int(np.nanmax(age_num))
else:
    age_min, age_max = 15, 40

age_from = int(params.get("age_from", age_min))
age_to   = int(params.get("age_to",   age_max))

age_range_slider = st.sidebar.slider(
    "Edad (rango)",
    min_value=age_min, max_value=age_max,
    value=(max(age_min, age_from), min(age_max, age_to)),
    key="filter_age_slider"
)
age_min_num = st.sidebar.number_input(
    "Edad mínima", min_value=age_min, max_value=age_max,
    value=int(age_range_slider[0]), step=1, key="filter_age_min"
)
age_max_num = st.sidebar.number_input(
    "Edad máxima", min_value=age_min, max_value=age_max,
    value=int(age_range_slider[1]), step=1, key="filter_age_max"
)
age_range = (int(min(age_min_num, age_max_num)), int(max(age_min_num, age_max_num)))

# ---------- 7) Minutos jugados (≥)
global_min = max(900, int(df["Min"].min())) if "Min" in df else 900
global_max = int(df["Min"].max()) if "Min" in df else 3420
default_min = int(np.clip(900, global_min, global_max))

min_sel_slider = st.sidebar.slider(
    "Minutos jugados (≥)",
    min_value=global_min, max_value=global_max,
    value=default_min, key="filter_mins_slider"
)
min_sel = st.sidebar.number_input(
    "Escribir minutos (≥)",
    min_value=global_min, max_value=global_max,
    value=int(min_sel_slider), step=30, key="filter_mins_num"
)

# ---------- Aplica filtros comunes (incluye Jugador/Equipo/Competición/Temporada/Rol/Edad)
mask_common = True
if players_sel: mask_common &= df["Player"].isin(players_sel)
if squads_sel:  mask_common &= df["Squad"].isin(squads_sel)
if comp:        mask_common &= df["Comp"].isin(comp)
if season:      mask_common &= df["Season"].isin(season)
if rol:         mask_common &= df["Rol_Tactico"].isin(rol)
if age_num.size:
    mask_common &= age_num.between(age_range[0], age_range[1])

# Base para Overview (histórico/actual se trata en sus subpestañas)
dff_base = df.loc[mask_common].copy()

# Con minutos (para Ranking / Comparador / Similares / Shortlist y para histórico por defecto)
mask = mask_common & ((df["Min"] >= min_sel) if "Min" in df else True)
dff = df.loc[mask].copy()

# Guarda estado en URL (añadimos jugador/equipo)
st.query_params.update({
    "players": players_sel, "squads": squads_sel,
    "comp": comp, "rol": rol, "season": season,
    "min": str(min_sel),
    "age_from": str(age_range[0]), "age_to": str(age_range[1]),
})

# --------- Construcción del subconjunto activo (dff_view) ----------
mask_common = True
if len(players_sel): mask_common &= df["Player"].isin(players_sel)
if len(squads_sel):  mask_common &= df["Squad"].isin(squads_sel)
if len(comp):        mask_common &= df["Comp"].isin(comp)
if len(rol):         mask_common &= df["Rol_Tactico"].isin(rol)
if len(season):      mask_common &= df["Season"].isin(season)
if age_num.size:     mask_common &= age_num.between(age_range[0], age_range[1])

mask_minutes = (df["Min"] >= min_sel) if "Min" in df else True
dff_view = df.loc[mask_common & mask_minutes].copy()

# Persistir en URL
st.query_params.update({
    "comp": comp, "rol": rol, "season": season,
    "min": str(min_sel),
    "age_from": str(age_range[0]), "age_to": str(age_range[1]),
})

# ===================== Métricas ==========================
gk_metrics = ["Save%", "PSxG+/-_per90", "PSxG_per90", "Saves_per90", "CS%", "Launch%"]
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
    else:
        st.info("Faltan columnas xG_per90 / Gls_per90 en el subconjunto actual.")

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

        # Default column sin flex (para que respete width/minWidth)
        gb.configure_default_column(sortable=True, filter=True, resizable=True, floatingFilter=True, flex=0)

        # Anchos iniciales + minWidth y Jugador fijado
        gb.configure_column(label("Player"), pinned="left", width=220, minWidth=220)
        gb.configure_column(label("Squad"), width=160, minWidth=160)
        gb.configure_column(label("Season"), width=120, minWidth=120)
        gb.configure_column(label("Rol_Tactico"), header_name=label("Rol_Tactico"), width=170, minWidth=170)
        gb.configure_column(label("Comp"), width=150, minWidth=150)
        gb.configure_column(label("Min"), width=110, minWidth=110)
        gb.configure_column(label("Age"), width=90, minWidth=90)

        # Paginación + sidebar
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
        gb.configure_side_bar()

        # Que NO auto-encaje todo: dejamos scroll horizontal y mantenemos widths
        gb.configure_grid_options(domLayout="normal", suppressHorizontalScroll=False)

        grid_options = gb.build()

        AgGrid(
            tabla_disp,
            gridOptions=grid_options,
            theme="streamlit",
            update_mode=GridUpdateMode.NO_UPDATE,
            columns_auto_size_mode=ColumnsAutoSizeMode.NO_AUTOSIZE,  # <- respeta width/minWidth
            fit_columns_on_grid_load=False,
            height=580,
            allow_unsafe_jscode=False,
        )
    except Exception:
        st.info("Para fijar columna/anchos usa `streamlit-aggrid` en `requirements.txt`. Mostrando tabla estándar.")
        st.dataframe(tabla_disp, use_container_width=True)

# ===================== COMPARADOR ========================
with tab_compare:
    stop_if_empty(dff_view)
    st.subheader("Comparador de jugadores (Radar)")
    players = dff_view["Player"].dropna().unique().tolist()
    cA, cB = st.columns(2)
    p1 = cA.selectbox("Jugador A", players, index=0 if players else None, key="pA")
    p2 = cB.selectbox("Jugador B", players, index=1 if len(players)>1 else 0, key="pB")

    radar_feats = st.multiselect(
        "Métricas para el radar (elige 4–8)",
        options=metrics_all,
        default=metrics_all[:6],
        key="feats",
        format_func=lambda c: label(c)
    )

    def radar(df_in, pA, pB, feats):
        S = df_in[feats].astype(float)
        S = normalize_0_1(S)
        A = S[df_in["Player"]==pA].mean(numeric_only=True).fillna(0)
        B = S[df_in["Player"]==pB].mean(numeric_only=True).fillna(0)
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=A.values, theta=[label(f) for f in feats], fill="toself", name=p1))
        fig.add_trace(go.Scatterpolar(r=B.values, theta=[label(f) for f in feats], fill="toself", name=p2))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True)
        return fig

    if p1 and p2 and radar_feats:
        st.plotly_chart(radar(dff_view, p1, p2, radar_feats), use_container_width=True)

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

# ===================== Footer ===========================
meta = next(DATA_DIR.glob("metadata_*.json"), None)
if meta and meta.exists():
    import json
    m = json.loads(meta.read_text(encoding="utf-8"))
    st.caption(f"📦 Dataset: {m.get('files',{}).get('parquet','parquet')} · "
               f"Filtros base: ≥{m.get('filters',{}).get('minutes_min',900)}′ · "
               f"Generado: {m.get('created_at','')}")


