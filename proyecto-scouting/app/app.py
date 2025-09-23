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

# ------------------- Helpers generales -------------------
def normalize_0_1(df_num: pd.DataFrame) -> pd.DataFrame:
    mn = df_num.min(axis=0); mx = df_num.max(axis=0)
    return (df_num - mn) / (mx - mn + 1e-9)

def _norm_season(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s).strip().replace("-", "/").replace("\\", "/")
    if "/" in s:
        parts = s.split("/")
        try:
            y1 = int(parts[0]); y2 = int(parts[1])
            if y2 >= 2000: y2 = y2 % 100
            s = f"{y1}/{y2:02d}"
        except: pass
    return s

def _season_key(s: str) -> int:
    """Convierte '2025/26' -> 202526 para ordenar de forma robusta."""
    if not s: return -1
    try:
        y1, y2 = s.split("/")
        return int(y1)*100 + int(y2)
    except:
        return -1

# Normaliza Season por si acaso
if "Season" in df.columns:
    df["Season"] = df["Season"].astype(str).map(_norm_season)

# ===================== Query params =======================
params = dict(st.query_params)
def _to_list(v): return [] if v is None else (v if isinstance(v, list) else [v])

comp_pre   = _to_list(params.get("comp"))
rol_pre    = _to_list(params.get("rol"))          # Rol_Tactico
season_pre = _to_list(params.get("season"))
min_pre    = int(params.get("min", 900))
age_from   = int(params.get("age_from", 15))
age_to     = int(params.get("age_to", 40))

# ===================== Filtros (sidebar) ==================
st.sidebar.header("Filtros")

comp_opts   = sorted(df["Comp"].dropna().unique())
rol_opts    = sorted(df["Rol_Tactico"].dropna().unique())
season_opts = sorted(df["Season"].dropna().unique(), key=_season_key)

# Detecta temporada "actual" como la más reciente
current_season = season_opts[-1] if season_opts else None

comp   = st.sidebar.multiselect("Competición", comp_opts, default=comp_pre)
rol    = st.sidebar.multiselect("Rol táctico (posición funcional)", rol_opts, default=rol_pre)
season = st.sidebar.multiselect("Temporada", season_opts, default=season_pre)

# Minutos: mínimo 900 (no se permite bajar de ahí) -> aplica a vistas generales y Ranking/Comparador
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

# --------- Máscaras de filtro (comunes) ----------
# Mask común (sin minutos) -> se usará en OVERVIEW (para construir ambas vistas)
mask_common = True
if age_num.size: mask_common &= age_num.between(age_range[0], age_range[1])
if comp:   mask_common &= df["Comp"].isin(comp)
if rol:    mask_common &= df["Rol_Tactico"].isin(rol)
if season: mask_common &= df["Season"].isin(season)
dff_base = df.loc[mask_common].copy()

# Mask con minutos -> se usa para el resto de pestañas y para histórico
mask = mask_common & ((df["Min"] >= min_sel) if "Min" in df else True)
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

# ===================== Tabs de nivel 1 ===================
tab_overview, tab_ranking, tab_compare, tab_similarity, tab_shortlist = st.tabs(
    ["📊 Overview", "🏆 Ranking", "🆚 Comparador", "🧬 Similares", "⭐ Shortlist"]
)

# ===================== Utilidad vacío ====================
def stop_if_empty(dfx):
    if len(dfx) == 0:
        st.warning("No hay jugadores que cumplan con estas condiciones de filtro. "
                   "Prueba a bajar el umbral de minutos, ampliar las edades o seleccionar más roles/temporadas.")
        st.stop()

# ===================== OVERVIEW ==========================
with tab_overview:
    # ---------- Sub-tabs: Histórico vs Temporada en curso ----------
    tab_hist, tab_cur = st.tabs(["📚 Histórico (≥900’)", "⏳ Temporada en curso"])

    # ---------------- HISTÓRICO ----------------
    with tab_hist:
        # Si el usuario seleccionó temporadas, usamos las seleccionadas excepto la actual
        selected_seasons = set(season) if season else set(df["Season"].dropna().unique())
        hist_seasons = [s for s in selected_seasons if s != current_season] if current_season else list(selected_seasons)

        df_hist = dff_base.copy()
        if hist_seasons:
            df_hist = df_hist[df_hist["Season"].isin(hist_seasons)]
        if "Min" in df_hist.columns:
            df_hist = df_hist[df_hist["Min"] >= 900]  # fijo a ≥900' para robustez histórica

        if df_hist.empty:
            st.warning("No hay jugadores históricos con ≥900′ en los filtros seleccionados.")
        else:
            # KPIs
            k1, k2, k3, k4 = st.columns(4, gap="large")
            with k1: st.metric("Jugadores (histórico)", f"{len(df_hist):,}")
            with k2: st.metric("Equipos", f"{df_hist['Squad'].nunique()}")
            with k3:
                try:
                    st.metric("Media edad", f"{pd.to_numeric(df_hist['Age'], errors='coerce').mean():.1f}")
                except: st.metric("Media edad", "—")
            with k4:
                med = int(df_hist["Min"].median()) if "Min" in df_hist else 0
                st.metric("Minutos medianos", f"{med:,}")

            # Scatter ofensivo
            if all(c in df_hist.columns for c in ["xG_per90","Gls_per90"]):
                fig = px.scatter(
                    df_hist, x="xG_per90", y="Gls_per90",
                    size=df_hist.get("Sh_per90", None),
                    color="Rol_Tactico", hover_data=["Player","Squad","Season"],
                    labels={"xG_per90":"xG por 90", "Gls_per90":"Goles por 90", "Rol_Tactico":"Rol táctico"},
                    template="plotly_dark", title="Productividad ofensiva (Histórico)"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Narrativa rápida (histórico)
            with st.expander("🗒️ Lectura rápida (histórico)", expanded=False):
                top_role = (df_hist["Rol_Tactico"].value_counts().idxmax()
                            if "Rol_Tactico" in df_hist and not df_hist.empty else "—")
                best_xg = df_hist.loc[df_hist["xG_per90"].idxmax()] if "xG_per90" in df_hist and df_hist["xG_per90"].notna().any() else None
                lines = [
                    f"- **Rol predominante:** {top_role}",
                    f"- **Edad media del conjunto:** {pd.to_numeric(df_hist['Age'], errors='coerce').mean():.1f}" if "Age" in df_hist else "- **Edad media:** —",
                ]
                if best_xg is not None:
                    lines.append(f"- **Máximo xG/90:** {best_xg['Player']} ({best_xg['Squad']}) — {best_xg['xG_per90']:.2f}")
                st.markdown("\n".join(lines))

    # ---------------- TEMPORADA EN CURSO ----------------
    with tab_cur:
        if current_season is None:
            st.info("No se pudo determinar la temporada actual.")
        else:
            # Solo datos de la temporada actual (sin imponer 900′)
            df_cur_all = dff_base[dff_base["Season"] == current_season].copy()

            # Control de minutos específico para la pestaña "en curso"
            if "Min" in df_cur_all.columns:
                st.info(f"Mostrando datos de **{current_season}** sin umbral de 900′. "
                        f"Puedes filtrar minutos específicamente para esta vista.")
                cur_min_default = 90 if df_cur_all["Min"].max() >= 90 else 0
                cur_min = st.slider("Minutos (≥) — solo para temporada en curso",
                                    min_value=0, max_value=int(df_cur_all["Min"].max()),
                                    value=int(cur_min_default), step=30, key="cur_min_slider")
                df_cur = df_cur_all[df_cur_all["Min"] >= cur_min].copy()
                if cur_min < 900:
                    st.warning("Estás viendo muestras <900′: interpretar con cautela (muestra parcial).")
            else:
                df_cur = df_cur_all.copy()

            if df_cur.empty:
                st.warning(f"No hay jugadores en {current_season} con los filtros y el umbral de minutos elegido.")
            else:
                # KPIs
                c1, c2, c3 = st.columns(3)
                c1.metric("Jugadores (en curso)", f"{len(df_cur):,}")
                c2.metric("Equipos", f"{df_cur['Squad'].nunique()}")
                try:
                    c3.metric("Media edad", f"{pd.to_numeric(df_cur['Age'], errors='coerce').mean():.1f}")
                except:
                    c3.metric("Media edad", "—")

                # Scatter ofensivo
                if all(c in df_cur.columns for c in ["xG_per90","Gls_per90"]):
                    fig_cur = px.scatter(
                        df_cur, x="xG_per90", y="Gls_per90",
                        size=df_cur.get("Sh_per90", None), color="Rol_Tactico",
                        hover_data=["Player","Squad"],
                        labels={"xG_per90":"xG por 90", "Gls_per90":"Goles por 90", "Rol_Tactico":"Rol táctico"},
                        template="plotly_dark", title=f"Productividad ofensiva ({current_season})"
                    )
                    st.plotly_chart(fig_cur, use_container_width=True)

                # Top minutos acumulados
                if "Min" in df_cur.columns:
                    top_min = df_cur.nlargest(10, "Min")
                    fig_min = px.bar(
                        top_min.sort_values("Min"),
                        x="Min", y="Player", color="Squad",
                        labels={"Min":"Minutos acumulados", "Player":"Jugador"},
                        template="plotly_dark", orientation="h",
                        title=f"⏱️ Top 10 por minutos acumulados ({current_season})"
                    )
                    st.plotly_chart(fig_min, use_container_width=True)

                # Radar por rol (media)
                roles_cols = [c for c in ["xG_per90","Gls_per90","xA_per90","KP_per90"] if c in df_cur.columns]
                if roles_cols:
                    avg_cur = df_cur.groupby("Rol_Tactico")[roles_cols].mean(numeric_only=True).reset_index()
                    if not avg_cur.empty and len(roles_cols) >= 3:
                        fig_radar = px.line_polar(
                            avg_cur.melt(id_vars="Rol_Tactico", var_name="Métrica", value_name="Valor"),
                            r="Valor", theta="Métrica", color="Rol_Tactico", line_close=True,
                            template="plotly_dark", title="Radar medio por rol táctico"
                        )
                        fig_radar.update_traces(fill="toself")
                        st.plotly_chart(fig_radar, use_container_width=True)

                # Narrativa rápida (en curso)
                with st.expander("🗒️ Lectura rápida (temporada en curso)", expanded=False):
                    top_role = (df_cur["Rol_Tactico"].value_counts().idxmax()
                                if "Rol_Tactico" in df_cur and not df_cur.empty else "—")
                    max_min = df_cur.loc[df_cur["Min"].idxmax()] if "Min" in df_cur and df_cur["Min"].notna().any() else None
                    lines = [
                        f"- **Rol más presente hasta ahora:** {top_role}",
                        f"- **Edad media del grupo:** {pd.to_numeric(df_cur['Age'], errors='coerce').mean():.1f}" if "Age" in df_cur else "- **Edad media:** —",
                    ]
                    if max_min is not None:
                        lines.append(f"- **Más minutos acumulados:** {max_min['Player']} ({max_min['Squad']}) — {int(max_min['Min'])}′")
                    st.markdown("\n".join(lines))

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
