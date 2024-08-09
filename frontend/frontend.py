import streamlit as st
from PIL import Image
import requests
import os

# Set page to wide mode
st.set_page_config(layout="wide")

# Initialize session state to store file names and chat history
if 'file_names' not in st.session_state:
    st.session_state['file_names'] = []

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'selected_file' not in st.session_state:
    st.session_state['selected_file'] = None

# Function to get file extension
def get_file_extension(file_name):
    return file_name.split('.')[-1]

# Function to get the icon path based on the file extension
def get_icon_path(file_extension):
    icon_folder = "frontend\icons"
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
        ##DELETE##
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
st.subheader("Chat with Chatbot")
for msg in st.session_state.history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Chat input at the bottom
prompt = st.chat_input("Say something")

# Append chat input to chat history and handle file selection
if prompt:
    if st.session_state['selected_file']:
        st.session_state['history'].append({
            'role': 'user',
            'content': prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner('💡Thinking'):
            data = {
                'question': prompt,
                'file_name': st.session_state['selected_file']
            }

            # Corrected: Use params=data to send as query parameters
            response = requests.post("http://127.0.0.1:8000/ask-question/", params=data)

            if response.status_code == 200:
                answer = response.json().get("response", "No response")
            else:
                st.write("Error Details:", response.json())  # Print the error details
                answer = f"Error {response.status_code}: {response.text}"

            st.session_state['history'].append({
                'role': 'Assistant',
                'content': answer
            })

            with st.chat_message("Assistant"):
                st.markdown(answer)
    else:
        st.warning("Please select a file and enter a message.")