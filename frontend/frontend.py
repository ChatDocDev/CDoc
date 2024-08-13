import streamlit as st
from PIL import Image
import requests
import os

# Set page to wide mode
st.set_page_config(layout="wide")

# Initialize session state to store file names and chat history
if 'file_names' not in st.session_state:
    st.session_state['file_names'] = []

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'selected_files' not in st.session_state:
    st.session_state['selected_files'] = []

if 'deleted_files' not in st.session_state:
    st.session_state['deleted_files'] = set()

# Function to get file extension
def get_file_extension(file_name):
    return file_name.split('.')[-1]

# Function to get the icon path based on the file extension
def get_icon_path(file_extension):
    icon_folder = "icons"
    icon_path = os.path.join(icon_folder, f"{file_extension}.png")
    if not os.path.exists(icon_path):
        icon_path = os.path.join(icon_folder, "default.png")
    return icon_path

# File uploader and file list in the sidebar
st.sidebar.header("Upload and Select Files")
uploaded_file = st.sidebar.file_uploader("Upload a file", type=['pdf', 'doc', 'txt', 'csv', 'xlsx'], accept_multiple_files=True)

# Handle file uploads
if uploaded_file is not None:
    # Create a copy of the uploaded_file to iterate over, so we can clear it after
    files_to_process = uploaded_file.copy()
    for file in files_to_process:
        file_extension = get_file_extension(file.name)
        file_record = {
            "name": file.name,
            "type": file_extension,
            "icon": get_icon_path(file_extension),
        }

        # Send the file to the FastAPI backend
        files = {'file': file}
        response = requests.post("http://127.0.0.1:8000/process-file/", files=files)

        if response.status_code == 200:
            if file_record not in st.session_state.file_names and file_record["name"] not in st.session_state.deleted_files:
                st.toast(f'{file.name} is uploaded successfully', icon="✅")
                st.session_state.file_names.append(file_record)

        else:
            st.sidebar.error(f"File upload failed: {response.json().get('error', 'Unknown error')}")

    # Clear the file uploader after processing all files
    uploaded_file = None

# Display uploaded files with icons and delete button in the sidebar
if st.session_state.file_names:
    st.sidebar.subheader("Uploaded Files")
    # Filter out deleted files
    files_to_display = [file for file in st.session_state.file_names if file["name"] not in st.session_state.deleted_files]
    for index, file in enumerate(files_to_display):
        icon_path = file["icon"]
        file_name = file["name"]

        col1, col2, col3 = st.sidebar.columns([1, 4, 1])  # Adjust column widths as needed
        with col1:
            icon_image = Image.open(icon_path)
            col1.image(icon_image, width=20)
        with col2:
            file_name_style = "color: lightgray;"
            col2.markdown(f"<span style='{file_name_style}'>{file_name}</span>", unsafe_allow_html=True)
        with col3:
            if col3.button("🗑️", key=f"delete_{index}"):
                # Mark file as deleted
                st.session_state.deleted_files.add(file_name)
                response = requests.post("http://127.0.0.1:8000/delete-file/", params={"file_name": file_name})
                if response.status_code == 200:
                    st.toast(f'{file_name} is deleted successfully', icon="🗑️")
                    # Remove from file names list in session state
                    st.session_state.file_names = [file for file in st.session_state.file_names if file["name"] != file_name]
                else:
                    st.sidebar.error(f"Failed to delete file: {response.json().get('error', 'Unknown error')}")

# Multi-select in the sidebar with icons
if st.session_state.file_names:
    file_names_list = [file['name'] for file in st.session_state.file_names if file["name"] not in st.session_state.deleted_files]
    selected_files = st.sidebar.multiselect(
        "Select files",
        options=file_names_list
    )
    st.session_state['selected_files'] = selected_files

# Display chat history using st.chat_message
st.subheader("Chat with Documents")
for msg in st.session_state.chat_history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Chat input at the bottom
prompt = st.chat_input("Say something")

# Append chat input to chat history and handle file selection
if prompt:
    if st.session_state['selected_files']:
        st.session_state['chat_history'].append({
            'role': 'user',
            'content': prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare a placeholder for the streamed response
        assistant_message = st.chat_message("Assistant")
        response_text = ""
        with assistant_message:
            message_placeholder = st.empty()

        # Display Assistant response with streaming support
        data = {
            'question': prompt,
            'file_names': st.session_state['selected_files']
        }

        response = requests.post("http://127.0.0.1:8000/ask-question/", json=data, stream=True)

        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    response_text += chunk.decode('utf-8')
                    message_placeholder.markdown(response_text)
            
            # Append the full response to chat history
            st.session_state['chat_history'].append({
                'role': 'Assistant',
                'content': response_text
            })
        else:
            st.write("Error Details:", response.json())  # Print the error details
            answer = f"Error {response.status_code}: {response.text}"
    else:
        st.warning("Please select files and enter a message.")
