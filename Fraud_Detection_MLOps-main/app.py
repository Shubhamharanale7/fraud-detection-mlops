from Src.model import FraudPipeline, FeatureEngineering, Preprocessing, LogTransformer

import streamlit as st
from Pages.home import home_page
from Pages.about_model import about_page
from Pages.metrics_page import metrics_page
from Pages.about_me import about_me_page

# Register pages with st.Page
home_pg = st.Page(home_page, title="Home", icon="🏠", default=True)
about_pg = st.Page(about_page, title="About Model", icon="ℹ️")
metrics_pg = st.Page(metrics_page, title="Metrics", icon="📊")
about_me_pg = st.Page(about_me_page, title="About Me", icon="👨‍💻")

# Navigation
pg = st.navigation([home_pg, about_pg, metrics_pg, about_me_pg])
pg.run()