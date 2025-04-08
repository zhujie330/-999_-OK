import streamlit as st


def load_css():
    st.markdown(
        """
        <style>
        .navbar {
            overflow: hidden;
            background-color: #333;
        }

        .navbar a {
            float: left;
            display: block;
            color: #f2f2f2;
            text-align: center;
            padding: 14px 16px;
            text-decoration: none;
        }

        .navbar a:hover {
            background-color: #ddd;
            color: black;
        }

        .navbar a.active {
            background-color: #04AA6D;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def main_sidebar():
    st.set_page_config(page_title="Streamlit 导航示例")

    menu = ["首页", "产品中心", "解决方案", "新闻中心", "AI治理研究院", "加入我们"]
    choice = st.sidebar.selectbox("菜单", menu)

    if choice == "首页":
        st.title("首页")
        st.write("欢迎来到首页。")
    elif choice == "产品中心":
        st.title("产品中心")
        st.write("欢迎来到产品中心。")
    elif choice == "解决方案":
        st.title("解决方案")
        st.write("欢迎来到解决方案页面。")
    elif choice == "新闻中心":
        st.title("新闻中心")
        st.write("欢迎来到新闻中心页面。")
    elif choice == "AI治理研究院":
        st.title("AI治理研究院")
        st.write("欢迎来到AI治理研究院页面。")
    elif choice == "加入我们":
        st.title("加入我们")
        st.write("欢迎来到加入我们页面。")


def main_navbar():
    load_css()

    st.markdown(
        """
        <div class="navbar">
            <a class="active" href="#home">首页</a>
            <a href="#products">产品中心</a>
            <a href="#solutions">解决方案</a>
            <a href="#news">新闻中心</a>
            <a href="#ai_research">AI治理研究院</a>
            <a href="#join_us">加入我们</a>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 你可以在这里添加更多逻辑来根据选定的菜单项显示不同的内容
    st.title("欢迎")
    st.write("这是一个示例页面。")


if __name__ == '__main__':
    st.sidebar.title("选择导航栏类型")
    nav_type = st.sidebar.radio("导航栏类型", ["侧边栏", "顶部导航栏"])

    if nav_type == "侧边栏":
        main_sidebar()
    else:
        main_navbar()
import streamlit as st


def load_css():
    st.markdown(
        """
        <style>
        .navbar {
            overflow: hidden;
            background-color: #333;
        }

        .navbar a {
            float: left;
            display: block;
            color: #f2f2f2;
            text-align: center;
            padding: 14px 16px;
            text-decoration: none;
        }

        .navbar a:hover {
            background-color: #ddd;
            color: black;
        }

        .navbar a.active {
            background-color: #04AA6D;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def main_sidebar():
    st.set_page_config(page_title="Streamlit 导航示例")

    menu = ["首页", "产品中心", "解决方案", "新闻中心", "AI治理研究院", "加入我们"]
    choice = st.sidebar.selectbox("菜单", menu)

    if choice == "首页":
        st.title("首页")
        st.write("欢迎来到首页。")
    elif choice == "产品中心":
        st.title("产品中心")
        st.write("欢迎来到产品中心。")
    elif choice == "解决方案":
        st.title("解决方案")
        st.write("欢迎来到解决方案页面。")
    elif choice == "新闻中心":
        st.title("新闻中心")
        st.write("欢迎来到新闻中心页面。")
    elif choice == "AI治理研究院":
        st.title("AI治理研究院")
        st.write("欢迎来到AI治理研究院页面。")
    elif choice == "加入我们":
        st.title("加入我们")
        st.write("欢迎来到加入我们页面。")


def main_navbar():
    load_css()

    st.markdown(
        """
        <div class="navbar">
            <a class="active" href="#home">首页</a>
            <a href="#products">产品中心</a>
            <a href="#solutions">解决方案</a>
            <a href="#news">新闻中心</a>
            <a href="#ai_research">AI治理研究院</a>
            <a href="#join_us">加入我们</a>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 你可以在这里添加更多逻辑来根据选定的菜单项显示不同的内容
    st.title("欢迎")
    st.write("这是一个示例页面。")


if __name__ == '__main__':
    st.sidebar.title("选择导航栏类型")
    nav_type = st.sidebar.radio("导航栏类型", ["侧边栏", "顶部导航栏"])

    if nav_type == "侧边栏":
        main_sidebar()
    else:
        main_navbar()
