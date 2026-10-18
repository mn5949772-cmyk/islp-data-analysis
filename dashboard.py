
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="ISLP Dataset Explorer", layout="wide")
st.title("ISLP Dataset Explorer")
st.write("Interactive exploration of Auto, College, and Boston teaching datasets. Use the filter to update all visuals.")

@st.cache_data
def load_data():
    base = Path(__file__).parent
    auto = pd.read_csv(base / "Auto.csv").replace("?", np.nan)
    for c in ["mpg","cylinders","displacement","horsepower","weight","acceleration","year","origin"]:
        auto[c] = pd.to_numeric(auto[c], errors="coerce")
    auto["origin_label"] = auto["origin"].map({1:"American",2:"European",3:"Japanese"})
    college = pd.read_csv(base / "College.csv")
    boston = pd.read_csv(base / "Boston.csv")
    boston["chas_label"] = boston["chas"].map({0:"Does not bound Charles River",1:"Bounds Charles River"})
    return auto, college, boston

auto, college, boston = load_data()
dataset = st.sidebar.selectbox("Choose dataset", ["Auto", "College", "Boston"])
st.sidebar.caption("The dataset selector is the shared filter for the four visuals below.")

if dataset == "Auto":
    options = ["All origins", "American", "European", "Japanese"]
    chosen = st.sidebar.selectbox("Filter by origin", options)
    d = auto.copy() if chosen == "All origins" else auto[auto["origin_label"] == chosen].copy()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(d, x="mpg", nbins=20, title="Fuel economy distribution"), use_container_width=True)
    with c2:
        st.plotly_chart(px.scatter(d, x="weight", y="mpg", color="origin_label", hover_data=["name"], title="Weight vs fuel economy"), use_container_width=True)
    with c1:
        st.plotly_chart(px.box(d, x="cylinders", y="mpg", title="Fuel economy by cylinder count"), use_container_width=True)
    yearly = d.groupby("year", as_index=False).agg(mean_mpg=("mpg","mean"), mean_weight=("weight","mean"))
    yearly["model_year"] = yearly["year"] + 1900
    with c2:
        st.plotly_chart(px.line(yearly, x="model_year", y="mean_mpg", markers=True, title="Mean mpg by model year"), use_container_width=True)
    st.caption(f"Filtered sample: {len(d)} cars. The shared origin filter updates all four charts.")

elif dataset == "College":
    options = ["All colleges", "Private", "Public"]
    chosen = st.sidebar.selectbox("Filter by college type", options)
    d = college.copy() if chosen == "All colleges" else college[college["Private"] == ("Yes" if chosen == "Private" else "No")].copy()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(d, x="Outstate", nbins=25, title="Out-of-state tuition distribution"), use_container_width=True)
    with c2:
        st.plotly_chart(px.scatter(d, x="Outstate", y="Apps", size="Enroll", color="Private", hover_data=["Grad.Rate"], title="Tuition vs applications (bubble size = enrollment)"), use_container_width=True)
    with c1:
        st.plotly_chart(px.box(d, x="Private", y="Grad.Rate", title="Graduation rate by college type"), use_container_width=True)
    with c2:
        st.plotly_chart(px.scatter(d, x="PhD", y="Outstate", color="Private", hover_data=["Apps"], title="Faculty PhD percentage vs tuition"), use_container_width=True)
    st.caption(f"Filtered sample: {len(d)} colleges. The shared college-type filter updates all four charts.")

else:
    options = ["All tracts", "Bounds Charles River", "Does not bound Charles River"]
    chosen = st.sidebar.selectbox("Filter by Charles River status", options)
    d = boston.copy() if chosen == "All tracts" else boston[boston["chas_label"] == chosen].copy()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(d, x="medv", nbins=25, title="Median home-value distribution"), use_container_width=True)
    with c2:
        st.plotly_chart(px.scatter(d, x="rm", y="medv", color="chas_label", hover_data=["lstat"], title="Rooms vs median home value"), use_container_width=True)
    with c1:
        st.plotly_chart(px.scatter(d, x="lstat", y="medv", title="Lower-status population vs median home value"), use_container_width=True)
    with c2:
        st.plotly_chart(px.box(d, x="chas_label", y="medv", title="Median home value by river adjacency"), use_container_width=True)
    st.caption(f"Filtered sample: {len(d)} census tracts. The shared river-status filter updates all four charts.")

st.divider()
st.subheader("How to interpret")
st.write("These charts are descriptive. Associations and visible patterns do not establish causal effects. Boston is a historical teaching dataset, not a current housing-market dataset.")
