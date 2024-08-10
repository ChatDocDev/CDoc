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

if 'selected_file' not in st.session_state:
    st.session_state['selected_file'] = None

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
st.sidebar.header("Upload and Select File")
uploaded_file = st.sidebar.file_uploader("Upload a file", type=['pdf', 'doc', 'txt', 'csv', 'xlsx'])

# Send uploaded file to backend and save file info in session state
if uploaded_file is not None:
    file_extension = get_file_extension(uploaded_file.name)
    file_record = {
        "name": uploaded_file.name,
        "type": file_extension,
        "icon": get_icon_path(file_extension),
    }

    # Send the file to the FastAPI backend
    files = {'file': uploaded_file}
    response = requests.post("http://127.0.0.1:8000/process-file/", files=files)

    if response.status_code == 200:
        if file_record not in st.session_state.file_names:
            st.session_state.file_names.append(file_record)
    else:
        st.sidebar.error(f"File upload failed: {response.json().get('error', 'Unknown error')}")

# Display uploaded files with icons and delete button in the sidebar
if st.session_state.file_names:
    st.sidebar.subheader("Uploaded Files")
    for index, file in enumerate(st.session_state.file_names):
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
                # Remove the file from the session state
                del st.session_state.file_names[index]
                st.experimental_rerun()

# Multi-select in the sidebar with icons
if st.session_state.file_names:
    file_names_list = [file['name'] for file in st.session_state.file_names]
    selected_file = st.sidebar.selectbox(
        "Select a file",
        options=file_names_list
    )
    st.session_state['selected_file'] = selected_file

# Display chat history using st.chat_message
st.subheader("Chat with Document")
for msg in st.session_state.chat_history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Chat input at the bottom
prompt = st.chat_input("Say something")

# Append chat input to chat history and handle file selection
if prompt:
    if st.session_state['selected_file']:
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
            'file_name': st.session_state['selected_file']
        }

        response = requests.post("http://127.0.0.1:8000/ask-question/", params=data, stream=True)

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
        st.warning("Please select a file and enter a message.")
