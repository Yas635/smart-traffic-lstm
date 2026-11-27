import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from config import DATA_PATH, FEATURES, MODEL_PATH, SCALER_PATH, SEQUENCE_LENGTH
from data_processing import load_and_enrich
from predict import build_feature_row, predict_next
from train_model import load_checkpoint, load_model

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Traffic Solutions",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card { background:#1e1e2e; border-radius:12px; padding:16px 20px; }
    .stMetric label { font-size:0.8rem !important; color:#aaa !important; }
</style>
""", unsafe_allow_html=True)

# ── Data & Model ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Chargement des données…")
def load_data() -> pd.DataFrame:
    df = load_and_enrich(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["date"] = df["timestamp"].dt.date
    return df

@st.cache_resource(show_spinner="Chargement du modèle…")
def get_model():
    return load_model()

@st.cache_data(show_spinner=False)
def get_speed_stats():
    with open(SCALER_PATH, "rb") as fh:
        return pickle.load(fh)

if not DATA_PATH.exists():
    st.error(f"Fichier de données introuvable : {DATA_PATH}. Voir data/README.md.")
    st.stop()

df = load_data()

try:
    model = get_model()
    speed_stats = get_speed_stats()
    model_loaded = True
    checkpoint = load_checkpoint()
except Exception:
    model_loaded = False
    checkpoint = {}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🚦 Smart Traffic")
    st.markdown("---")

    date_min = df["date"].min()
    date_max = df["date"].max()
    date_range = st.date_input(
        "Période d'analyse",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        d1, d2 = date_range
        df_view = df[(df["date"] >= d1) & (df["date"] <= d2)].copy()
    else:
        df_view = df.copy()

    st.markdown("---")
    if model_loaded:
        st.success("Modèle LSTM chargé ✅")
        st.caption(f"Accuracy : **{checkpoint.get('accuracy', 0):.3f}**")
        st.caption(f"F1-score : **{checkpoint.get('f1', 0):.3f}**")
        st.caption(f"ROC-AUC  : **{checkpoint.get('auc', 0):.3f}**")
    else:
        st.warning("Modèle non trouvé — entraînez d'abord avec `python train_model.py`")

    st.markdown("---")
    st.caption("Sources : PeMS Dataset · PyTorch LSTM")

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🚦 Smart Traffic Solutions")
st.markdown("##### Analyse et prédiction de la congestion à partir de données historiques PeMS")
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────────────────────

total = len(df_view)
n_cong = int(df_view["congestion"].sum())
rate = n_cong / total * 100 if total else 0
avg_speed = df_view["speed"].mean()
peak_hour = int(df_view.groupby("hour")["congestion"].mean().idxmax()) if total else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Données analysées", f"{total:,}", "mesures 5 min")
col2.metric("Taux de congestion", f"{rate:.1f} %", f"{n_cong:,} événements")
col3.metric("Vitesse moyenne", f"{avg_speed:.1f} km/h")
col4.metric("Heure de pointe", f"{peak_hour:02d}h", "taux max")

st.markdown("---")

# ── Row 1 : vitesse + congestion horaire ─────────────────────────────────────

col_l, col_r = st.columns([2, 1])

with col_l:
    st.subheader("📈 Évolution de la vitesse")
    sample = df_view.set_index("timestamp")["speed"]
    # Sous-échantillonne si trop de points (performance)
    if len(sample) > 5000:
        sample = sample.iloc[::len(sample) // 2000]
    fig_speed = px.line(
        sample.reset_index(), x="timestamp", y="speed",
        labels={"speed": "Vitesse (km/h)", "timestamp": ""},
        color_discrete_sequence=["#4f8ef7"],
    )
    fig_speed.add_hline(
        y=60, line_dash="dot", line_color="red",
        annotation_text="Seuil congestion (60 km/h)",
        annotation_position="bottom right",
    )
    fig_speed.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig_speed, use_container_width=True)

with col_r:
    st.subheader("🕐 Congestion horaire")
    hourly = df_view.groupby("hour")["congestion"].mean().reset_index()
    hourly.columns = ["hour", "rate"]
    fig_bar = px.bar(
        hourly, x="hour", y="rate",
        labels={"hour": "Heure", "rate": "Taux"},
        color="rate",
        color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
        range_color=[0, 1],
    )
    fig_bar.update_layout(height=300, margin=dict(t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Row 2 : heatmap jour×heure ────────────────────────────────────────────────

st.subheader("🗓️ Heatmap congestion — Jour × Heure")
pivot = df_view.groupby(["day_of_week", "hour"])["congestion"].mean().reset_index()
pivot_table = pivot.pivot(index="day_of_week", columns="hour", values="congestion").fillna(0)

day_labels = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
pivot_table.index = [day_labels[i] for i in pivot_table.index]

fig_heat = px.imshow(
    pivot_table,
    labels=dict(x="Heure", y="Jour", color="Taux congestion"),
    color_continuous_scale="RdYlGn_r",
    aspect="auto",
    zmin=0, zmax=1,
)
fig_heat.update_layout(height=280, margin=dict(t=10, b=10))
st.plotly_chart(fig_heat, use_container_width=True)

# ── Row 3 : Prédiction LSTM ───────────────────────────────────────────────────

st.markdown("---")
st.subheader("🤖 Prédiction LSTM")

if model_loaded:
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("**Prédire un moment précis**")
        h = st.slider("Heure", 0, 23, 8)
        m = st.select_slider("Minute", options=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], value=0)

        if st.button("Prédire", type="primary"):
            target_min = h * 60 + m
            df["min_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
            match = df[df["min_of_day"] == target_min]

            if match.empty or match.index[-1] < SEQUENCE_LENGTH:
                st.warning("Données insuffisantes pour cette heure.")
            else:
                idx = match.index[-1]
                seq = df[FEATURES].values[idx - SEQUENCE_LENGTH: idx].astype("float32")
                x = torch.tensor(seq).unsqueeze(0)
                with torch.no_grad():
                    prob = model(x).item()

                st.metric(f"Probabilité à {h:02d}h{m:02d}", f"{prob*100:.1f} %")
                if prob > 0.5:
                    st.error(f"🚨 Embouteillage probable ({prob*100:.0f}%)")
                else:
                    st.success(f"✅ Trafic fluide ({prob*100:.0f}%)")

    with col_b:
        st.markdown("**Prochains instants (à partir des dernières données)**")
        n_steps = st.slider("Nombre de pas à prédire (×5 min)", 1, 24, 6)

        if st.button("Simuler les prochains instants"):
            with st.spinner("Prédiction en cours…"):
                results = predict_next(n_steps)

            res_df = pd.DataFrame(results)
            res_df["timestamp"] = res_df["timestamp"].dt.strftime("%H:%M")
            res_df["prob_pct"] = (res_df["prob"] * 100).round(1)
            res_df["état"] = res_df["congestion"].map({0: "✅ Fluide", 1: "🚨 Congestion"})

            fig_fut = px.bar(
                res_df, x="timestamp", y="prob_pct",
                color="prob_pct",
                color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
                range_color=[0, 100],
                labels={"timestamp": "Heure", "prob_pct": "Probabilité (%)"},
            )
            fig_fut.add_hline(y=50, line_dash="dot", line_color="white")
            fig_fut.update_layout(height=250, margin=dict(t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig_fut, use_container_width=True)
else:
    st.info("Entraînez le modèle avec `python train_model.py` pour activer les prédictions.")

# ── Row 4 : Carte ─────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🗺️ Carte illustrative (données simulées)")

selected_hour = st.slider("Heure affichée sur la carte", 0, 23, 8, key="map_hour")

np.random.seed(42)
n_pts = 40
center_lat, center_lon = 33.5731, -7.5898

hour_data = df_view[df_view["hour"] == selected_hour]
cong_rate = float(hour_data["congestion"].mean()) if len(hour_data) else 0.3

map_df = pd.DataFrame({
    "lat": center_lat + np.random.uniform(-0.05, 0.05, n_pts),
    "lon": center_lon + np.random.uniform(-0.05, 0.05, n_pts),
    "congestion": np.random.choice(
        [0, 1], size=n_pts,
        p=[1 - cong_rate, cong_rate]
    ),
})
map_df["color"] = map_df["congestion"].map({0: "#2ecc71", 1: "#e74c3c"})
map_df["état"] = map_df["congestion"].map({0: "Fluide", 1: "Congestion"})

fig_map = px.scatter_mapbox(
    map_df, lat="lat", lon="lon",
    color="état",
    color_discrete_map={"Fluide": "#2ecc71", "Congestion": "#e74c3c"},
    size_max=12,
    zoom=13,
    mapbox_style="carto-darkmatter",
    title=f"Simulation à {selected_hour:02d}h (taux de congestion des données : {cong_rate*100:.0f}%)",
)
fig_map.update_layout(height=420, margin=dict(t=30, b=0))
st.plotly_chart(fig_map, use_container_width=True)

st.caption("Positions et états tirés au hasard autour de Casablanca : démonstration visuelle, pas des mesures réelles.")
