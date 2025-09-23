import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Scouting LaLiga", layout="wide")

# ---------- Carga de datos ----------
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

# ---------- Filtros ----------
st.sidebar.header("Filtros")
comp = st.sidebar.multiselect("Competición", sorted(df["Comp"].dropna().unique()))
rol  = st.sidebar.multiselect("Rol táctico", sorted(df["Rol_Tactico"].dropna().unique()))
pos  = st.sidebar.multiselect("Posición", sorted(df["Pos"].dropna().unique()))

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

# ---------- Métricas ----------
gk_metrics = ["Save%", "PSxG+/-_per90", "PSxG_per90", "Saves_per90", "CS%", "Launch%"]
out_metrics = [
    "Gls_per90","xG_per90","NPxG_per90","Sh_per90","SoT_per90","G/SoT_per90",
    "xA_per90","KP_per90","GCA90_per90","SCA_per90","PrgP_per90","PrgC_per90",
    "Carries_per90","Cmp%","Cmp_per90","Tkl+Int_per90","Int_per90","Recov_per90"
]
is_gk_view = (pos == ["GK"]) or (len(pos)==1 and "GK" in pos)
metrics = gk_metrics if is_gk_view else out_metrics
metrics = [m for m in metrics if m in dff.columns]

st.title("⚽ Scouting LaLiga")
st.caption("Métricas por 90’ + porcentajes (0–100). Filtros por competición, rol y posición.")

# ---------- Ranking ----------
st.subheader("Ranking")
metric_to_rank = st.selectbox("Métrica para ordenar", metrics, index=0 if metrics else None)
topn = st.slider("Top N", 5, 50, 20)
cols_show = ["Player","Squad","Season","Pos","Rol_Tactico","Comp","Min","Age"] + metrics
tabla = dff[cols_show].sort_values(metric_to_rank, ascending=False).head(topn)
st.dataframe(tabla, use_container_width=True)

# ---------- Comparador (radar) ----------
st.subheader("Comparador de jugadores")
players = dff["Player"].dropna().unique().tolist()
colA, colB = st.columns(2)
p1 = colA.selectbox("Jugador A", players, index=0 if players else None)
p2 = colB.selectbox("Jugador B", players, index=1 if len(players)>1 else 0)

def radar(df_in, pA, pB, feats):
    S = df_in[feats].astype(float)
    S = (S - S.min()) / (S.max() - S.min() + 1e-9)
    A = S[df_in["Player"]==pA].mean(numeric_only=True).fillna(0)
    B = S[df_in["Player"]==pB].mean(numeric_only=True).fillna(0)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=A.values, theta=feats, fill='toself', name=pA))
    fig.add_trace(go.Scatterpolar(r=B.values, theta=feats, fill='toself', name=pB))
    fig.update_layout(polar=dict(radialaxis=d_
