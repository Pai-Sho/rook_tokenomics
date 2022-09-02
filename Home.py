import streamlit as st

from PIL import Image

icon = Image.open("images/KeeperDAO_Logo_Icon_White.png")
st.set_page_config(page_title="ROOK Tokenomics", page_icon=icon)

st.markdown(
    "### **NOTE:** This model only attempts to simulate the things we have some control over, with somewhat reasonable methods. It is not an accurate representation of reality, and should only be used to compare the _relative_ effects of changing these parameters in a vacuum."
)
