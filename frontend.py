"""
Serves index.html as a full-viewport Streamlit page.
Run with:  streamlit run frontend.py
"""
import pathlib
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="GPU Cloud Business Model",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Strip all Streamlit chrome so the HTML fills the page
st.markdown("""
<style>
  #MainMenu, header[data-testid="stHeader"], footer { display: none !important; }
  .block-container { padding: 0 !important; max-width: 100% !important; }
  section[data-testid="stSidebar"] { display: none !important; }
  [data-testid="stAppViewContainer"] { overflow: hidden; }
</style>
""", unsafe_allow_html=True)

html_path = pathlib.Path(__file__).parent / "index.html"
components.html(html_path.read_text(encoding="utf-8"), height=940, scrolling=False)
