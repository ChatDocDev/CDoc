import streamlit as st
from PIL import Image
import requests
import os

# Set page to wide mode
st.set_page_config(layout="wide")

# Initialize session state to store file names, chat history, selected files, and deleted files
if 'file_names' not in st.session_state:
    st.session_state['file_names'] = []

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'selected_files' not in st.session_state:
    st.session_state['selected_files'] = []

if 'deleted_files' not in st.session_state:
    st.session_state['deleted_files'] = set()  # Use set for faster lookup

# Function to get file extension
def get_file_extension(file_name):
    """
    Get the file extension from the given file name.

    Args:
        file_name (str): The name of the file.

    Returns:
        str: The file extension.
    """
    return file_name.split('.')[-1]

# Function to get the icon path based on the file extension
def get_icon_path(file_extension):
    """
    Get the path to the icon image based on the file extension.

    Args:
        file_extension (str): The file extension.

    Returns:
        str: The path to the icon image.
    """
    icon_folder = "icons"
    icon_path = os.path.join(icon_folder, f"{file_extension}.png")
    if not os.path.exists(icon_path):
        icon_path = os.path.join(icon_folder, "default.png")
    return icon_path

# Sidebar for file uploading and selection
st.sidebar.header("Upload and Select File(s)")

# Streamlit form for file uploads
with st.sidebar.form(key='upload_form',clear_on_submit=True):
    """
    Streamlit form for uploading files.
    """
    uploaded_files = st.file_uploader("Upload a file(s)", type=['pdf', 'docx', 'txt', 'csv', 'xlsx'], accept_multiple_files=True)
    submit_button = st.form_submit_button(label='Upload')

# Handle file uploads
if submit_button and uploaded_files:
    """
    Handles the file upload process. Uploads files to the backend and updates session state.
    """
    for file in uploaded_files:
        file_extension = get_file_extension(file.name)
        file_record = {
            "name": file.name,
            "type": file_extension,
            "icon": get_icon_path(file_extension),
        }
        
        # Remove from deleted files if re-uploaded
        st.session_state.deleted_files.discard(file_record["name"])
        
        # Only upload if not already in session state
        if file_record["name"] not in [f["name"] for f in st.session_state.file_names]:
            # Send file to backend
            files = {'file': file}
            response = requests.post("http://127.0.0.1:8000/process-file/", files=files)
            
            if response.status_code == 200:
                st.toast(f'{file.name} is uploaded successfully', icon="✅")
                st.session_state.file_names.append(file_record)
            else:
                st.sidebar.error(f"File upload failed: {response.json().get('error', 'Unknown error')}")

# Display uploaded files with icons and delete button in the sidebar
if st.session_state.file_names:
    """
    Multi-select widget for selecting files from the uploaded files. Updates the session state with selected files.
    """
    st.sidebar.subheader("Uploaded Files")
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
                file_name = file["name"]
                st.session_state.deleted_files.add(file_name)
                
                # Send delete request to backend
                response = requests.post("http://127.0.0.1:8000/delete-file/", params={"file_name": file_name})
                
                if response.status_code == 200:
                    st.toast(f'{file_name} is deleted successfully', icon="🗑️")
                    st.session_state.file_names = [file for file in st.session_state.file_names if file["name"] != file_name]
                else:
                    st.sidebar.error(f"Failed to delete file: {response.json().get('error', 'Unknown error')}")
                
                # Re-run the script to refresh the sidebar
                st.rerun()

# Multi-select in the sidebar with icons
if st.session_state.file_names:
    file_names_list = [file['name'] for file in st.session_state.file_names if file["name"] not in st.session_state.deleted_files]
    selected_files = st.sidebar.multiselect(
        "Select file(s)",
        options=file_names_list
    )
    st.session_state['selected_files'] = selected_files

# Display chat history using st.chat_message
st.subheader("CDOC: Chat with your Documents")
for msg in st.session_state.chat_history:
    """
    Display chat history with messages from the user and the assistant.
    """
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Chat input at the bottom
prompt = st.chat_input("Select file(s) before asking something")

# Append chat input to chat history and handle file selection
if prompt:
    if st.session_state['selected_files']:
        """
        Handle chat input by appending it to chat history and sending a request to the backend for a response.
        """
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
        st.warning("Please select file(s) and enter a message.")