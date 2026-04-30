import re
import time

import streamlit as st
from shared.constants import TIME_LIMIT_MAPPING

from frontend.constants import (
    IGNORED_MESSAGES,
    STAGE_COMPLETE_SIGNAL,
    STAGE_MESSAGES,
    STAGE_TO_STATUS_BAR,
    STATUS_BAR_STAGE,
    SUCCESS_MESSAGE,
)


def parse_model_path(log):
    """
    Extract the model/artifacts path from the log text.

    Args:
        log (str): The log text containing the model path

    Returns:
        str or None: The extracted model path or None if not found
    """
    # Try AutogluonModels path pattern (quoted)
    pattern = r'"([^"]*AutogluonModels[^"]*)"'
    match = re.search(pattern, log)
    if match:
        return match.group(1)

    # Try artifacts path pattern for FEDOT (e.g., "saved at /path/to/artifacts")
    # Match "saved at" followed by the rest of the line as the path
    artifacts_pattern = r"(?:saved at)\s+(.+)$"
    match = re.search(artifacts_pattern, log, re.IGNORECASE)
    if match:
        # Strip any trailing whitespace or punctuation
        path = match.group(1).strip()
        return path

    return None


def show_log_line(line):
    """
    Show log line based on prefix and Rich syntax
    - Lines starting with WARNING: → st.warning
    - Other lines → st.markdown
    """
    if "INFO:" in line:
        parts = line.split(":", 2)
        if len(parts) >= 3:
            line = parts[2]
    if any(message in line for message in STAGE_COMPLETE_SIGNAL):
        return st.success(line)
    elif line.startswith("WARNING:"):
        return st.warning(line)
    return st.markdown(line)


def get_stage_from_log(log_line):
    for message, stage in STAGE_MESSAGES.items():
        if message in log_line:
            return stage
    return None


def show_logs():
    """
    Display logs and task status when task is finished.
    """
    if st.session_state.logs:
        status_container = st.empty()
        if st.session_state.return_code == 0:
            status_container.success(SUCCESS_MESSAGE)
        else:
            status_container.error("Ошибка при выполнении. Смотрите логи для деталей")
        tab1, tab2 = st.tabs(["Сообщения", "Логи"])
        with tab1:
            for stage, logs in st.session_state.stage_container.items():
                if logs:
                    with st.status(stage, expanded=False, state="complete"):
                        for log in logs:
                            show_log_line(log)
        with tab2:
            log_container = st.empty()
            log_container.text_area("Логи выполнения", st.session_state.logs, height=400)


def format_log_line(line):
    """
    Format log lines by removing ANSI escape codes, formatting markdown syntax,
    and cleaning up process-related information.

    Args:
        line (str): Raw log line to be formatted.

    Returns:
        str: Formatted log line with:
    """
    line = re.sub(r"\x1B\[1m(.*?)\x1B\[0m", r"**\1**", line)
    line = re.sub(r"^#", r"\\#", line)
    line = re.sub(r"\033\[\d+m", "", line)
    line = re.sub(r"^\s*\(\w+ pid=\d+\)\s*", "", line)
    return line


def process_realtime_logs(line):
    """
    Handles the real-time processing of log lines, updating the UI state,
    managing progress bars, and displaying status updates in the  interface.

    Args:  line (str): A single line from the log stream to process.
    """
    if any(ignored_msg in line for ignored_msg in IGNORED_MESSAGES):
        return
    stage = get_stage_from_log(line)
    if stage:
        if stage != st.session_state.current_stage:
            if st.session_state.current_stage:
                st.session_state.stage_status[st.session_state.current_stage].update(
                    state="complete",
                )
            st.session_state.current_stage = stage
        if stage not in st.session_state.stage_status:
            st.session_state.stage_status[stage] = st.status(stage, expanded=False)

    if st.session_state.current_stage:
        if "model training complete" in line.lower():
            st.session_state.show_remaining_time = False
        with st.session_state.stage_status[st.session_state.current_stage]:
            if "model training starts" in line.lower() and not st.session_state.show_remaining_time:
                st.session_state.progress_bar = st.progress(0, text="Время обучения:")
                st.session_state.show_remaining_time = True
                st.session_state.elapsed_time = time.time() - st.session_state.start_time
                st.session_state.remaining_time = (
                    TIME_LIMIT_MAPPING[st.session_state.time_limit] - st.session_state.elapsed_time
                )
                st.session_state.start_model_train_time = time.time()
            if st.session_state.show_remaining_time:
                st.session_state.elapsed_time = time.time() - st.session_state.start_model_train_time
                progress_bar = st.session_state.progress_bar
                current_time = min(st.session_state.elapsed_time, st.session_state.remaining_time)
                progress = current_time / st.session_state.remaining_time
                time_ratio = f"Время обучения: | ({progress:.1%})"
                progress_bar.progress(progress, text=time_ratio)
            if not st.session_state.show_remaining_time:
                st.session_state.stage_container[st.session_state.current_stage].append(line)
                show_log_line(line)


def messages():
    """
    Handles the streaming of log messages from a subprocess, updates the UI with progress
    indicators, and manages the display of different training stages.
    processes ANSI escape codes, formats markdown, and updates various progress indicators.

    """
    if st.session_state.process is not None:
        process = st.session_state.process
        st.session_state.logs = ""
        progress = st.progress(0)
        status_container = st.empty()
        st.session_state.start_time = time.time()
        status_container.info("Выполняется...")
        for line in process.stdout:
            print(line, end="")
            line = format_log_line(line)
            st.session_state.logs += line
            # Check for model path indicators (works for both AutoGluon and FEDOT)
            model_path_indicators = [
                "tabularpredictor saved",
                "model saved",
                "pipeline saved",
                "artifacts",
            ]
            if any(indicator in line.lower() for indicator in model_path_indicators):
                model_path = parse_model_path(line)
                if model_path:
                    st.session_state.model_path = model_path
            # Check for completion signals
            if "prediction complete!" in line.lower():
                status_container.success(SUCCESS_MESSAGE)
                progress.progress(100)
                process_realtime_logs(line)
                st.toast("Готово! 🎉", icon="✅")
                # Keep draining stdout so post-prediction lines
                # (e.g. artifacts save path) are still captured.
                for remaining_line in process.stdout:
                    remaining_line = format_log_line(remaining_line)
                    st.session_state.logs += remaining_line
                    if any(indicator in remaining_line.lower() for indicator in model_path_indicators):
                        model_path = parse_model_path(remaining_line)
                        if model_path:
                            st.session_state.model_path = model_path
                break
            else:
                for stage_msg, progress_value in STATUS_BAR_STAGE.items():
                    if stage_msg.lower() in line.lower():
                        progress.progress(progress_value / 100)
                        status_container.info(STAGE_TO_STATUS_BAR[stage_msg])
                        break
            process_realtime_logs(line)
            if st.session_state.current_stage and st.session_state.current_stage in st.session_state.stage_status:
                st.session_state.stage_status[st.session_state.current_stage].update(
                    state="running",
                )
