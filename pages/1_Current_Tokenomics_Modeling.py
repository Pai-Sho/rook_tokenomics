import streamlit as st

st.set_page_config(layout="wide")

with open("markdown/Modeling.md") as homepage:
    st.markdown(homepage.read())
