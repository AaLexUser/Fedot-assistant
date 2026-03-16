import os
import shutil
import uuid

import pandas as pd
import streamlit as st

BASE_DATA_DIR = "./user_data"


def get_user_data_dir():
    """
    Get or create a unique directory for the current user session.

    Returns:
        str: The path to the user's data directory.
    """
    if "user_data_dir" not in st.session_state:
        unique_dir = st.session_state.user_session_id
        st.session_state.user_data_dir = os.path.join(BASE_DATA_DIR, unique_dir)
        os.makedirs(st.session_state.user_data_dir, exist_ok=True)
    return st.session_state.user_data_dir


def get_user_session_id():
    """
    Get or generate a unique user session ID.

    Returns:
        str: A unique identifier for the current user session.
    """
    if "user_session_id" not in st.session_state:
        st.session_state.user_session_id = str(uuid.uuid4())
    return st.session_state.user_session_id


def generate_output_filename():
    """
    Generate a unique output filename based on the user session ID and current timestamp.

    Returns:
        str: A unique filename for the output CSV file.
    """
    user_data_dir = get_user_data_dir()
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    directory_path = os.path.join(user_data_dir, timestamp)
    os.makedirs(directory_path, exist_ok=True)
    output_filepath = os.path.join(directory_path, "predictions.csv")
    return output_filepath


def generate_output_file():
    """
    Generate and store the output file after task completion.
    """
    if st.session_state.return_code == 0:
        output_filename = st.session_state.output_filename
        if output_filename:
            df = pd.read_csv(output_filename)
            st.session_state.output_file = df
        else:
            st.error(f"CSV file not found: {output_filename}")


def clear_directory(directory):
    """
    Before saving the files to the user's directory, clear the directory first
    but keep the 'description.txt' file.

    Args:
       directory (str): Directory path to be cleared.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isdir(file_path) or filename == "description.txt":
            continue
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def save_uploaded_file(file, directory):
    """
    Save an uploaded file to the specified directory.

    Args:
        file (UploadedFile): The file uploaded by the user.
        directory (str): The directory to save the file in.
    """
    file_path = os.path.join(directory, file.name)
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())


def save_all_files(user_data_dir):
    """
    When the task starts to run, save all the user's uploaded files to user's directory

    Args:
        user_data_dir (str): The directory path where user's files will be saved.
    """
    clear_directory(user_data_dir)
    for file_name, file_data in st.session_state.uploaded_files.items():
        save_uploaded_file(file_data["file"], user_data_dir)


def create_zip_file(model_path):
    """
    Create a zip file of the model directory

    Args:
        model_path (str): Path to the model directory

    Returns:
        str: Path to the created zip file
    """
    if not os.path.exists(model_path):
        return None
    zip_path = f"{model_path}.zip"
    shutil.make_archive(model_path, "zip", model_path)
    return zip_path


def generate_model_file():
    if st.session_state.model_path:
        zip_path = create_zip_file(st.session_state.model_path)
        st.session_state.zip_path = zip_path
