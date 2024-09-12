# %% packages

import streamlit as st

# %%
st.title("About")

# %% page navigation
# Create a sidebar with options for navigation
st.sidebar.title('Navigation')

st.sidebar.page_link(page="app.py", label="Home")
st.sidebar.page_link(page="pages/about.py", label="About")