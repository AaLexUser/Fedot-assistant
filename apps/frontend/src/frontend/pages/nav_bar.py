import streamlit as st


def nav_bar():
    """
    Show the top navigation bar
    """
    st.markdown(
        """
    <nav class="navbar navbar-expand-sm navbar-light bg-white fixed-top" style="padding-left: 81px;">
        <div class="navbar-nav">
          <a class="nav-item nav-link" href="#zapustit-fedot-llm" style="color: #18A0FB;">Запустить</a>
          <a class="nav-item nav-link" href="#preview-dataset" style="color: #18A0FB;">Датасет</a>
          <a class="nav-item nav-link disabled" href="https://github.com/AaLexUser/Fedot-assistant" style="color: #18A0FB;">Github</a>
        </div>
    </nav>
    """,
        unsafe_allow_html=True,
    )


def main():
    nav_bar()


if __name__ == "__main__":
    main()
