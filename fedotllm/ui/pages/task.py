import copy
import os
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import psutil
import requests
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from fedotllm.constants import (
    AUTOML_ENGINE_OPTIONS,
    BASE_DATA_DIR,
    BASE_URL_MAPPING,
    CAPTIONS,
    DATASET_OPTIONS,
    EXTRACT_DIR,
    INITIAL_STAGE,
    LLM_MAPPING,
    LLM_OPTIONS,
    LOCAL_ZIP_PATH,
    PRESET_DEFAULT_CONFIG,
    PRESET_MAPPING,
    PRESET_OPTIONS,
    S3_URL,
    TIME_LIMIT_MAPPING,
    TIME_LIMIT_OPTIONS,
)
from fedotllm.ui.file_uploader import (
    description_file_uploader,
    file_uploader,
    save_description_file,
)
from fedotllm.ui.log_processor import messages, show_logs
from fedotllm.ui.utils import (
    generate_model_file,
    generate_output_file,
    generate_output_filename,
    get_user_data_dir,
    get_user_session_id,
    save_all_files,
)

os.makedirs(BASE_DATA_DIR, exist_ok=True)


def update_config_overrides():
    """
    Update the configuration overrides based on the current session state.
    """
    config_overrides = []
    if st.session_state.time_limit:
        config_overrides.append(
            f"time_limit={TIME_LIMIT_MAPPING[st.session_state.time_limit]}"
        )
    if st.session_state.llm:
        config_overrides.append(f"llm.model={LLM_MAPPING[st.session_state.llm]}")
        config_overrides.append(
            f"llm.base_url={BASE_URL_MAPPING[st.session_state.llm]}"
        )

    if st.session_state.automl_engine:
        config_overrides.append(f"automl.enabled={st.session_state.automl_engine}")

    st.session_state.config_overrides = config_overrides


def store_value(key):
    """
    Store a value in the session state.

    Args:
        key (str): The key to store the value under in the session state.
    """
    st.session_state[key] = st.session_state["_" + key]
    if key == "preset":
        preset_config = PRESET_DEFAULT_CONFIG.get(st.session_state.preset)
        st.session_state["time_limit"] = preset_config.get("time_limit")
        st.session_state["feature_generation"] = preset_config.get("feature_generation")


def load_value(key):
    """
    Load a value from the session state into a temporary key.

    Args:
        key (str): The key to load the value from in the session state.
    """
    st.session_state["_" + key] = st.session_state[key]


def config_preset():
    load_value("preset")
    st.selectbox(
        "Пресет",
        placeholder="Выберите пресет",
        options=PRESET_OPTIONS,
        key="_preset",
        on_change=store_value,
        args=["preset"],
        label_visibility="collapsed",
    )


def config_automl_engine():
    load_value("automl_engine")
    st.selectbox(
        "Automl Engine",
        placeholder="Automl Engine",
        options=AUTOML_ENGINE_OPTIONS,
        key="_automl_engine",
        on_change=store_value,
        args=["automl_engine"],
        label_visibility="collapsed",
    )


@st.fragment
def config_time_limit():
    load_value("time_limit")
    st.selectbox(
        "Ограничение по времени",
        placeholder="Выберите время",
        options=TIME_LIMIT_OPTIONS,
        key="_time_limit",
        on_change=store_value,
        args=["time_limit"],
        label_visibility="collapsed",
    )


@st.fragment
def config_llm():
    st.selectbox(
        "Модель LLM",
        placeholder="Выберите модель",
        options=LLM_OPTIONS,
        key="_llm",
        on_change=store_value,
        args=["llm"],
        label_visibility="collapsed",
    )


@st.fragment
def config_feature_generation():
    load_value("feature_generation")
    checkbox = st.checkbox(
        "Генерация признаков (бета)",
        key="_feature_generation",
        on_change=store_value,
        args=["feature_generation"],
    )
    # Show warning if checkbox is checked
    if checkbox:
        st.warning(
            "Генерация признаков — экспериментальная функция. Установите дополнительные пакеты: "
            "'pip install -r requirements.txt'",
            icon="⚠️",
        )


def store_value_and_save_file(key):
    store_value(key)
    save_description_file(st.session_state.task_description)


@st.fragment
def display_description():
    load_value("task_description")
    st.text_area(
        label="Описание датасета",
        placeholder="Опишите задачу: цель, признаки, что предсказываем...",
        key="_task_description",
        on_change=store_value_and_save_file,
        args=["task_description"],
        height=300,
    )


@st.fragment
def show_output_download_button(data, file_name):
    st.download_button(
        label="💾&nbsp;&nbsp;Скачать предсказания",
        data=data,
        file_name=file_name,
        mime="text/csv",
    )


def show_cancel_task_button():
    try:
        if st.button(
            "⏹️&nbsp;&nbsp;Стоп",
            on_click=toggle_cancel_state,
            key="cancel_task",
        ):
            p = st.session_state.process
            print("Останавливаем задачу...")
            p.terminate()
            p.wait()
            st.session_state.process = None
            st.session_state.pid = None
            print("Задача остановлена")
            st.session_state.task_running = False
            st.session_state.show_remaining_time = False
            st.session_state.output_filename = None
            st.session_state.zip_path = None
            st.rerun()
    except psutil.NoSuchProcess:
        st.session_state.task_running = False
        st.session_state.process = None
        st.session_state.pid = None
        st.error("Нет запущенных задач")
    except Exception as e:
        st.session_state.task_running = False
        st.session_state.process = None
        st.session_state.pid = None
        st.error(f"Ошибка: {e}")


def run_assistant_process(data_dir):
    """
    Run the FedotLLM with the specified configuration and data directories.

    This function constructs the command to run the FedotLLM, including
    any config overrides, and starts the process. The process object and output
    filename are stored in the session state.

    Args:
        config_dir (str): The path to the configuration directory.
        data_dir (str): The path to the data directory.
    """
    output_filename = generate_output_filename()
    command = ["fedotllm", "run", data_dir]
    if st.session_state.preset:
        command.extend(["--presets", PRESET_MAPPING[st.session_state.preset]])
    if st.session_state.config_overrides:
        command.extend(
            ["--config_overrides", ",".join(st.session_state.config_overrides)]
        )
    command.extend(["--output-filename", output_filename])
    st.session_state.output_file = None
    st.session_state.output_filename = output_filename
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        st.session_state.process = process
        st.session_state.pid = process.pid
    except Exception as e:
        print(f"An error occurred: {e}")


def download_model_button():
    """
    Create and display a download button for the log file.
    """
    if st.session_state.zip_path and st.session_state.task_running is False:
        zip_path = st.session_state.zip_path
        if zip_path and os.path.exists(zip_path):
            with open(zip_path, "rb") as f:
                st.download_button(
                    label="⬇️&nbsp;&nbsp;Скачать модель",
                    data=f,
                    file_name=os.path.basename(zip_path),
                    mime="application/zip",
                )


def download_log_button():
    """
    Create and display a download button for the log file.
    """
    if st.session_state.logs and st.session_state.task_running is False:
        st.download_button(
            label="📥&nbsp;&nbsp;Скачать логи",
            data=st.session_state.logs,
            file_name="logs.txt",
            mime="text/plain",
        )


def download_output_button():
    """
    Create and display a download button for the output file.
    """
    if (
        st.session_state.output_file is not None
        and st.session_state.task_running is False
    ):
        output_file = st.session_state.output_file
        output_filename = st.session_state.output_filename
        final_name = os.path.basename(output_filename)
        show_output_download_button(output_file.to_csv(index=False), final_name)


def toggle_running_state():
    st.session_state.task_running = True
    st.session_state.task_canceled = False
    st.session_state.current_stage = None
    st.session_state.stage_container = copy.deepcopy(INITIAL_STAGE)
    st.session_state.stage_status = {}
    st.session_state.increment_time = 0


def toggle_cancel_state():
    st.session_state.task_canceled = True


def run_button():
    """
    Create and handle the "Run" button for starting the task.
    """
    if st.button(
        label="🔘&nbsp;&nbsp;Запустить!",
        key="run_task",
        on_click=toggle_running_state,
        disabled=st.session_state.task_running,
    ):
        if st.session_state.selected_dataset == "Пример датасета":
            if st.session_state.sample_dataset_dir is not None:
                sample_data_dir = st.session_state.sample_dataset_dir
                run_assistant_process(sample_data_dir)
            else:
                st.warning("Выберите пример датасета")
                st.session_state.task_running = False
        elif st.session_state.selected_dataset == "Загрузить свой":
            user_data_dir = get_user_data_dir()
            if st.session_state.uploaded_files:
                save_all_files(user_data_dir)
                run_assistant_process(user_data_dir)
            else:
                st.warning("Сначала загрузите файлы")
                st.session_state.task_running = False


def show_cancel_container():
    """
    Display a cancellation message when the task is cancelled.
    """
    status_container = st.empty()
    status_container.info("Задача остановлена")


def set_description():
    """
    Set up the description section of the user interface.

    This function calls helper functions to display a description and set up
    a file uploader for the description
    """
    if st.session_state.selected_dataset == "Загрузить свой":
        display_description()
        description_file_uploader()
    elif st.session_state.selected_dataset == "Пример датасета":
        load_value("sample_description")
        st.text_area(
            label="Описание примера",
            placeholder="Описание задачи...",
            key="_sample_description",
            on_change=store_value,
            args=["sample_description"],
            height=300,
        )
    add_vertical_space(4)


def wait_for_process():
    """
    Wait for the process to complete and update session state based on return code.
    """
    if st.session_state.process is not None:
        process = st.session_state.process
        process.wait()
        st.session_state.task_running = False
        st.session_state.return_code = process.returncode
        st.session_state.process = None
        st.session_state.pid = None
        generate_output_file()
        generate_model_file()
        st.rerun()


def run_section():
    """
    Set up and display the 'Run FedotLLM' section of the user interface.
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
               Запустить FedotLLM
           </h1>
       """,
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4, col5 = st.columns([1, 10.9, 0.2, 10.9, 1], gap="large")
    with col4:
        dataset_selector()
    with col2:
        col11, col12 = st.columns(2)
        with col11:
            config_preset()
            config_time_limit()
        with col12:
            config_llm()
            config_automl_engine()
        set_description()
    with col3:
        st.html(
            """
                <div class="divider-vertical-line"></div>
                <style>
                    .divider-vertical-line {
                        border-left: 2px solid rgba(49, 51, 63, 0.2);
                        height: 590px;
                        margin: auto;
                    }
                </style>
            """
        )
    update_config_overrides()
    _, mid_pos, _ = st.columns([1, 22, 1], gap="large")
    with mid_pos:
        run_button()
        if st.session_state.task_running:
            show_cancel_task_button()
    _, mid_pos, _ = st.columns([1, 22, 1], gap="large")
    with mid_pos:
        if st.session_state.task_running:
            messages()
        elif not st.session_state.task_running and not st.session_state.task_canceled:
            show_logs()
        elif st.session_state.task_canceled:
            show_cancel_container()
    wait_for_process()
    _, download_pos, _ = st.columns([4, 2, 4])
    with download_pos:
        if not st.session_state.task_running:
            download_log_button()
            download_output_button()
            download_model_button()
    st.markdown("---", unsafe_allow_html=True)


def _extract_zip_strip_toplevel(zip_path: str, extract_dir: str):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()

        if members:
            first_parts = members[0].split("/")
            toplevel = first_parts[0] if first_parts else ""

            has_toplevel = all(m.startswith(toplevel + "/") for m in members if m.strip())

            if has_toplevel and toplevel:
                for member in members:
                    if member.strip():
                        relative_path = member[len(toplevel) + 1:]
                        if relative_path:
                            target_path = os.path.join(extract_dir, relative_path)
                            parent = os.path.dirname(target_path)
                            if parent:
                                os.makedirs(parent, exist_ok=True)
                            if not member.endswith("/"):
                                with zip_ref.open(member) as src, open(target_path, "wb") as dst:
                                    dst.write(src.read())
                return

        zip_ref.extractall(extract_dir)


def setup_local_dataset():
    """Download dataset from S3 to local directory"""
    try:
        Path(EXTRACT_DIR).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(EXTRACT_DIR):
            response = requests.get(S3_URL, stream=True)
            if response.status_code == 200:
                with open(LOCAL_ZIP_PATH, "wb") as f:
                    f.write(response.content)
                try:
                    _extract_zip_strip_toplevel(LOCAL_ZIP_PATH, EXTRACT_DIR)
                except zipfile.BadZipFile:
                    st.error("Не удалось распаковать архив")
                    return
                finally:
                    os.remove(LOCAL_ZIP_PATH)
            else:
                st.error(f"Ошибка загрузки: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при загрузке примера: {e}")
    except OSError as e:
        st.error(f"Ошибка при работе с файлами: {e}")
    except Exception as e:
        st.error(f"Неожиданная ошибка: {e}")


def get_sample_dataset_files(dataset_dir):
    """
    Get all files from the given sample dataset directory
    """
    st.session_state.sample_files = {}
    for file in dataset_dir.glob("*.csv"):
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        st.session_state.sample_files[file.name] = {"file": file, "df": df}


def get_available_datasets(sample_dataset_dir):
    """Get all subdirectories in the sample_dataset directory."""
    return [d.name for d in Path(sample_dataset_dir).iterdir() if d.is_dir()]


def sample_dataset_selector():
    sample_dataset_dir = EXTRACT_DIR
    sample_datasets = get_available_datasets(sample_dataset_dir)
    load_value("sample_dataset_selector")
    selected_dataset = st.selectbox(
        "Выберите датасет:",
        options=sample_datasets,
        key="_sample_dataset_selector",
        index=None,
        on_change=store_value,
        args=["sample_dataset_selector"],
    )
    if selected_dataset:
        selected_dataset_dir = Path(sample_dataset_dir) / selected_dataset
        st.session_state.sample_dataset_dir = selected_dataset_dir
        get_sample_dataset_files(selected_dataset_dir)
        description_file = selected_dataset_dir / "descriptions.txt"
        if description_file.exists():
            try:
                with open(description_file, "r", encoding="utf-8") as f:
                    st.session_state.sample_description = f.read()
            except Exception as e:
                st.error(f"Ошибка чтения описания: {e}")
        st.success(
            f"Выбран: ****{selected_dataset}****. Смотрите данные в разделе ****Датасет****"
        )
        st.success("Нажмите 🔘&nbsp;&nbsp;****Запустить****")


def dataset_selector():
    dataset_choice = st.radio(
        "Выбор датасета",
        key="dataset_choose",
        options=DATASET_OPTIONS,
        horizontal=True,
        captions=CAPTIONS,
        label_visibility="collapsed",
    )
    if dataset_choice == "Пример датасета":
        st.session_state.selected_dataset = "Пример датасета"
        sample_dataset_selector()
    elif dataset_choice == "Загрузить свой":
        st.session_state.selected_dataset = "Загрузить свой"
        file_uploader()


def main():
    setup_local_dataset()
    get_user_session_id()
    run_section()


if __name__ == "__main__":
    main()
