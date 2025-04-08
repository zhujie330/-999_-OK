# from st_pages import Page, Section, show_pages, add_page_title
#
# # Either this or add_indentation() MUST be called on each page in your
# # app to add indendation in the sidebar
# add_page_title()
#
# # Specify what pages should be shown in the sidebar, and what their titles and icons
# # should be
# show_pages(
#     [
#         Page("Home.py", "Home"),
#         Page("1_Deepfake_Detection.py", "Detection"),
#     ]
# )

import streamlit as st

# 设置页面标题和图标
# st.set_page_config(page_title="My Streamlit App", page_icon=":rocket:",layout="wide", initial_sidebar_state="auto")
st.sidebar.header("")
# 设置页面布局和主题
# st.set_page_config(layout="wide", initial_sidebar_state="auto")