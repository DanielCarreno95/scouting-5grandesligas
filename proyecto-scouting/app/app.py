# app/app.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
st.set_page_config(page_title="Scouting LaLiga", layout="wide")

# Encabezado bonito
st.markdown("""
<style>
.big-title {font-size:2.0rem; font-weight:800; margin:0 0 0.1rem 0;}
.subtle {color:#8A8F98; margin:0 0 1.0rem 0;}
</style>
<div class="big-title">⚽ Scouting LaLiga</div>
<p class="subtle">≥900′ · métricas por 90’ y % (0–100) · filtros por competición/rol/posición</p>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Utilidades
# ---------------------------------------------------------
def percentiles_por_pos(df_in, cols, pos_col="Pos"):
    df = df_in.copy()
    for c in cols:
        if c in df.columns:
            df[c + "_pct"] = df.groupby(pos_col)[c].transform(lambda s: s.rank(pct=True))
    return df

def score_compuesto(df_in, cols_peso):
    # cols_peso = {"xG_per90":0.4, "xA_per90":0.3, "Sh_per90":0.3}
    use_cols = [c for c in cols_peso.keys() if c in df_in.columns]
    if not use_cols: 
        return df_in.copy()
    base = df_in[use_cols].astype(float)
    scaler = MinMaxScaler()
    base = pd.DataFrame(scaler.fit_transform(base), columns=use_cols, index=df_in.index)
    score = sum(base[c] * w for c, w in cols_peso.items() if c in base.columns)
    out = df_in.copy()
    out["Score"] = score.round(3)
    return out

def normalize_0_1(arr, axis=0):
    mn = arr.min(axis=axis, keepdims=True)
    mx = arr.max(axis=axis, keepdims=True)
    return (arr - mn) / (mx - mn + 1e-9)

# ---------------------------------------------------------
# Parámetros de URL (para compartir estado)
# ---------------------------------------------------------
params = st.experimental_get_query_params()

# ---------------------------------------------------------
# Filtros
# ---------------------------------------------------------
st.sidebar.header("Filtros")

comp_opts = sorted(df["Comp"].dropna().unique())
rol_opts  = sorted(df["Rol_Tactico"].dropna().unique())
pos_opts  = sorted(df["Pos"].dropna().unique())

comp = st.sidebar.multiselect("Competición", comp_opts, default=[])
rol  = st.sidebar.multiselect("Rol táctico", rol_opts, default=[])
pos  = st.sidebar.multiselect("Posición", pos_opts, default=[])

min_min = int(df["Min"].min()) if "Min" in df else 0
min_max = int(df["Min"].max()) if "Min" in df else 3000
min_sel = st.sidebar.slider("Minutos (≥)", min_value=min_min, max_value=min_max, value=min_min)

age_num = pd.to_numeric(df.get("Age", pd.Series(dtype=float)), errors="coerce")
if age_num.size:
    age_min, age_max = int(np.nanmin(age_num)), int(np.nanmax(age_num))
else:
    age_min, age_max = 15, 40
age_range = st.sidebar.slider("Edad", min_value=age_min, max_value=age_max, value=(age_min, age_max))

mask = (df["Min"] >= min_sel)
if age_num.size:
    mask &= age_num.between(age_range[0], age_range[1])
if comp: mask &= df["Comp"].isin(comp)
if rol:  mask &= df["Rol_Tactico"].isin(rol)
if pos:  mask &= df["Pos"].isin(pos)

dff = df.loc[mask].copy()

# guarda el estado en la URL
st.experimental_set_query_params(
    comp=comp, rol=rol, pos=pos, min=min_sel, age_from=age_range[0], age_to=age_range[1]
)

# ---------------------------------------------------------
# Métricas base
# ---------------------------------------------------------
gk_metrics = ["Save%", "PSxG+/-_per90", "PSxG_per90", "Saves_per90", "CS%", "Launch%"]
out_metrics = [
    "Gls_per90","xG_per90","NPxG_per90","Sh_per90","SoT_per90","G/SoT_per90",
    "xA_per90","KP_per90","GCA90_per90","SCA_per90","PrgP_per90","PrgC_per90",
    "Carries_per90","Cmp%","Cmp_per90","Tkl+Int_per90","Int_per90","Recov_per90"
]
is_gk_view = (pos == ["GK"]) or (len(pos) == 1 and "GK" in pos)
metrics_all = gk_metrics if is_gk_view else out_metrics
metrics_all = [m for m in metrics_all if m in dff.columns]

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_overview, tab_ranking, tab_compare, tab_similarity, tab_shortlist = st.tabs(
    ["📊 Overview", "🏆 Ranking", "🆚 Comparador", "🧬 Similares", "⭐ Shortlist"]
)

# =============== OVERVIEW ===============
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jugadores", f"{len(dff):,}")
    c2.metric("Equipos", f"{dff['Squad'].nunique()}")
    try:
        c3.metric("Media edad", f"{pd.to_numeric(dff['Age'], errors='coerce').mean():.1f}")
    except Exception:
        c3.metric("Media edad", "—")
    c4.metric("Minutos medianos", f"{int(dff['Min'].median()) if 'Min' in dff else 0}")

    # Dispersión xG vs Goles
    if all(c in dff.columns for c in ["xG_per90","Gls_per90"]):
        fig = px.scatter(
            dff, x="xG_per90", y="Gls_per90", hover_name="Player",
            color=dff.get("Pos", None), size=dff.get("SoT_per90", None),
            title="Productividad ofensiva: xG/90 vs Goles/90"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay columnas xG_per90/Gls_per90 en el subconjunto actual.")

# =============== RANKING ===============
with tab_ranking:
    st.subheader("Ordena por métrica o define un Score compuesto")
    metric_to_rank = st.selectbox("Métrica para ordenar", metrics_all, index=0 if metrics_all else None)

    st.caption("Opcional: define un score (pesos que sumen 1). Ejemplo: xG_per90:0.4,xA_per90:0.3,Sh_per90:0.3")
    pesos_txt = st.text_input("Pesos", "")
    cols_peso = {}
    if pesos_txt.strip():
        for kv in pesos_txt.split(","):
            if ":" in kv:
                k, v = kv.split(":")
                k = k.strip()
                try:
                    cols_peso[k] = float(v)
                except:
                    pass
        cols_peso = {k:v for k, v in cols_peso.items() if k in metrics_all}
        if cols_peso:
            total = sum(cols_peso.values())
            if total == 0:
                cols_peso = {}
            elif abs(total - 1.0) > 1e-6:
                cols_peso = {k: v/total for k, v in cols_peso.items()}
                st.info("Los pesos no sumaban 1 → normalizados automáticamente.")

    dfr = dff.copy()
    dfr = percentiles_por_pos(dfr, metrics_all, pos_col="Pos")

    if cols_peso:
        dfr = score_compuesto(dfr, cols_peso)
        metric_to_rank = "Score"

    topn = st.slider("Top N", 5, 100, 20)
    cols_show = ["Player","Squad","Season","Pos","Rol_Tactico","Comp","Min","Age"] + metrics_all
    if "Score" in dfr.columns:
        cols_show = ["Score"] + cols_show

    tabla = dfr[cols_show].sort_values(metric_to_rank, ascending=False).head(topn)
    st.dataframe(tabla, use_container_width=True)

# =============== COMPARADOR ===============
with tab_compare:
    st.subheader("Comparador de jugadores (Radar)")
    players = dff["Player"].dropna().unique().tolist()
    colA, colB = st.columns(2)
    p1 = colA.selectbox("Jugador A", players, index=0 if players else None, key="pA")
    p2 = colB.selectbox("Jugador B", players, index=1 if len(players)>1 else 0, key="pB")

    radar_feats = st.multiselect("Métricas para el radar (elige 4–8)", metrics_all, default=metrics_all[:6], key="feats")

    def radar(df_in, pA, pB, feats):
        S = df_in[feats].astype(float)
        S = normalize_0_1(S.to_numpy(), axis=0)
        S = pd.DataFrame(S, columns=feats, index=df_in.index)
        A = S[df_in["Player"]==pA].mean(numeric_only=True).fillna(0)
        B = S[df_in["Player"]==pB].mean(numeric_only=True).fillna(0)
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=A.values, theta=feats, fill="toself", name=pA))
        fig.add_trace(go.Scatterpolar(r=B.values, theta=feats, fill="toself", name=pB))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True)
        return fig

    if p1 and p2 and radar_feats:
        st.plotly_chart(radar(dff, p1, p2, radar_feats), use_container_width=True)

# =============== SIMILARES ===============
with tab_similarity:
    st.subheader("Jugadores similares (cosine similarity)")
    feats_sim = st.multiselect("Selecciona 6–12 métricas", metrics_all, default=metrics_all[:8], key="sim_feats")
    target = st.selectbox("Jugador objetivo", dff["Player"].dropna().unique().tolist())

    if feats_sim and target:
        X = dff[feats_sim].astype(float).fillna(0.0).to_numpy()
        X = normalize_0_1(X, axis=0)
        # vector del jugador objetivo
        idx = dff.index[dff["Player"]==target][0]
        v = X[dff.index.get_loc(idx)]
        # coseno
        from numpy.linalg import norm
        sims = (X @ v) / (norm(X, axis=1)*norm(v) + 1e-9)

        out = dff[["Player","Squad","Season","Pos","Rol_Tactico","Comp","Min","Age"]].copy()
        out["similarity"] = sims
        out = out.sort_values("similarity", ascending=False).head(25)
        st.dataframe(out, use_container_width=True)

# =============== SHORTLIST ===============
with tab_shortlist:
    st.subheader("Shortlist (lista de seguimiento)")
    if "shortlist" not in st.session_state:
        st.session_state.shortlist = []

    to_add = st.multiselect("Añadir jugadores", dff["Player"].dropna().unique().tolist())
    if st.button("➕ Agregar seleccionados"):
        st.session_state.shortlist = sorted(set(st.session_state.shortlist) | set(to_add))

    to_remove = st.multiselect("Eliminar de shortlist", st.session_state.shortlist)
    if st.button("🗑️ Eliminar seleccionados"):
        st.session_state.shortlist = [p for p in st.session_state.shortlist if p not in set(to_remove)]

    sh = dff[dff["Player"].isin(st.session_state.shortlist)]
    st.dataframe(sh, use_container_width=True)

    st.download_button(
        "⬇️ Descargar shortlist (CSV)",
        data=sh.to_csv(index=False).encode("utf-8-sig"),
        file_name="shortlist_scouting.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------
# Footer de trazabilidad
# ---------------------------------------------------------
meta = next(DATA_DIR.glob("metadata_*.json"), None)
if meta and meta.exists():
    import json
    m = json.loads(meta.read_text(encoding="utf-8"))
    st.caption(
        f"📦 Dataset: {m.get('files',{}).get('parquet', 'parquet')} · "
        f"Filtros base: ≥{m.get('filters',{}).get('minutes_min',900)}′ · "
        f"Generado: {m.get('created_at','')}"
    )
