import streamlit as st
from PIL import Image
import os

# Initialize session state to store file names and chat history
if 'file_names' not in st.session_state:
    st.session_state['file_names'] = []

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

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

# Function to get the file path (assuming files are stored in 'uploads' folder)
def get_file_path(file_name):
    uploads_folder = "uploads"
    return os.path.join(uploads_folder, file_name)

# Ensure 'uploads' directory exists
os.makedirs("uploads", exist_ok=True)

# File uploader
uploaded_file = st.file_uploader("Upload a file", type=['pdf','doc','txt','csv', 'xlsx'])

# Save uploaded file names and paths
if uploaded_file is not None:
    file_extension = get_file_extension(uploaded_file.name)
    file_path = get_file_path(uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    file_record = {
        "name": uploaded_file.name,
        "type": file_extension,
        "icon": get_icon_path(file_extension),
        "path": file_path
    }
    
    if file_record not in st.session_state.file_names:
        st.session_state.file_names.append(file_record)

# Function to create an icon and file name display
def display_icon(file_record, container, highlight=False):
    icon_path = file_record["icon"]
    file_name = file_record["name"]
    
    col1, col2 = container.columns([1, 4])  # Adjust column widths as needed

    with col1:
        if os.path.exists(icon_path):
            icon_image = Image.open(icon_path)
            col1.image(icon_image, width=20)
        else:
            default_icon_path = os.path.join("icons", "default.png")
            default_icon_image = Image.open(default_icon_path)
            col1.image(default_icon_image, width=20)

    with col2:
        if highlight:
            file_name_style = "color: white;"
        else:
            file_name_style = "color: lightgray;"
            
        col2.markdown(f"<span style='{file_name_style}'>{file_name}</span>", unsafe_allow_html=True)


# Display uploaded files with icons in the main section
st.header("Uploaded Files")
if st.session_state.file_names:
    for file in st.session_state.file_names:
        display_icon(file, st)

# Multi-select in the sidebar with icons
st.sidebar.header("Select Files to Display")
if st.session_state.file_names:
    file_names_list = [file['name'] for file in st.session_state.file_names]
    selected_files = st.sidebar.multiselect(
        "Select files",
        options=file_names_list,
        default=file_names_list
    )

    for file in st.session_state.file_names:
        is_selected = file['name'] in selected_files
        display_icon(file, st.sidebar, highlight=is_selected)

# Display chat history
st.subheader("Chat with Chatbot")
if st.session_state['chat_history']:
    for chat in st.session_state['chat_history']:
        st.write(chat)

# Create a chat input bar at the bottom
chat_input = st.text_input("Type your message here:", key="chat_input")

# Append chat input to chat history
if st.button("Send"):
    if chat_input:
        st.session_state['chat_history'].append(f"You: {chat_input}")
        st.session_state['chat_history'].append(f"Chatbot: {get_chatbot_response(chat_input)}")
        st.experimental_rerun()  # To refresh the chat history display

# Function to simulate chatbot response
def get_chatbot_response(user_input):
    # This is a simple placeholder response. Replace with your chatbot logic.
    return f"Response to: {user_input}"