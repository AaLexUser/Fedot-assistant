import os

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder


def get_user_uploaded_files():
    files_name = []
    if st.session_state.uploaded_files is not None:
        uploaded_files = st.session_state.uploaded_files
        files_name = list(uploaded_files.keys())
    return files_name


def get_sample_files():
    files_name = []
    if st.session_state.sample_files is not None:
        sample_files = st.session_state.sample_files
        files_name = list(sample_files.keys())
    return files_name


@st.fragment
def preview_dataset():
    """
    Displays a preview of the uploaded dataset in the Streamlit app.
    """
    st.markdown(
        """
        <h1 style='
            font-weight: light;
            padding-left: 20px;
            padding-right: 20px;
            margin-left:60px;
            font-size: 2em;
        '>
            Датасет
        </h1>
    """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 22, 1])
    with col2:
        file_options = []
        if st.session_state.selected_dataset == "Загрузить свой":
            file_options = get_user_uploaded_files()
        elif st.session_state.selected_dataset == "Пример датасета":
            file_options = get_sample_files()
        if st.session_state.output_file is not None:
            file_options.append(os.path.basename(st.session_state.output_filename))
        if not file_options:
            st.info("Файлы не найдены", icon="ℹ️")
            return
        selected_file = st.selectbox(
            "Файл для просмотра",
            options=file_options,
            index=None,
            placeholder="Выберите файл",
            label_visibility="collapsed",
        )
        if not st.session_state.uploaded_files and not st.session_state.sample_files:
            st.info("Файлы не найдены", icon="ℹ️")
            return
        if selected_file is not None:
            st.markdown(
                f"""
            <div class="file-view-bar">
                <span class="file-view-label">Просмотр:</span> {selected_file}
            </div>
            """,
                unsafe_allow_html=True,
            )
            if st.session_state.output_filename and selected_file == os.path.basename(
                st.session_state.output_filename
            ):
                output_file = st.session_state.output_file
                gb = GridOptionsBuilder.from_dataframe(output_file)
                gb.configure_pagination()
                gridOptions = gb.build()
                AgGrid(
                    output_file,
                    gridOptions=gridOptions,
                    enable_enterprise_modules=False,
                )
            elif st.session_state.selected_dataset == "Загрузить свой":
                gb = GridOptionsBuilder.from_dataframe(
                    st.session_state.uploaded_files[selected_file]["df"]
                )
                gb.configure_pagination()
                gridOptions = gb.build()
                AgGrid(
                    st.session_state.uploaded_files[selected_file]["df"],
                    gridOptions=gridOptions,
                    enable_enterprise_modules=False,
                )
            elif st.session_state.selected_dataset == "Пример датасета":
                gb = GridOptionsBuilder.from_dataframe(
                    st.session_state.sample_files[selected_file]["df"]
                )
                gb.configure_pagination()
                gridOptions = gb.build()
                AgGrid(
                    st.session_state.sample_files[selected_file]["df"],
                    gridOptions=gridOptions,
                    enable_enterprise_modules=False,
                )


def main():
    preview_dataset()


if __name__ == "__main__":
    main()
