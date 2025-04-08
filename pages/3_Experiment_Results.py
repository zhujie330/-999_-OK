import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Deepfake Detection", page_icon="📈")
st.sidebar.header("📈Deepfake Detection")

st.write("# Demo for Deepfake Detection📝")

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

c = (
   alt.Chart(chart_data)
   .mark_circle()
   .encode(x="a", y="b", size="c", color="c", tooltip=["a", "b", "c"])
)

st.altair_chart(c, use_container_width=True)