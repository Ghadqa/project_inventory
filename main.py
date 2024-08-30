import streamlit as st
from functions import show_compare  # Import the pre-defined function from functions

# Set up the page configuration
st.set_page_config(page_title="Inventory Dashboard", page_icon=":bar_chart:", layout="wide")

# Inject custom CSS

st.markdown(
    """
    <style>
    /* Background color for the entire app */
    .reportview-container {
        background-color: #fefefe;  /* Light background color for the whole page */
    }
    
    /* Sidebar background color */
    .sidebar .sidebar-content {
        background-color: #c8102e;  /* Red color for the sidebar */
        color: #ffffff;  /* White text in the sidebar */
    }

    /* Sidebar title color */
    .sidebar .sidebar-title {
        color: #ffffff;
    }

    /* Main header styling */
    h1 {
        color: #c8102e;  /* Red color for the main header */
    }

    /* Page title styling */
    .css-18e3th9 {
        color: #c8102e;
        font-size: 24px;
    }

    /* Customize buttons */
    .stButton button {
        background-color: #c8102e;  /* Red background for buttons */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px;
    }
    .stButton button:hover {
        background-color: #a60a24;  /* Darker red for button hover effect */
    }

    /* Customize text input */
    .stTextInput input {
        border-radius: 5px;
        border: 1px solid #c8102e;  /* Red border for text input fields */
    }
    </style>
    """,
    unsafe_allow_html=True
)
def main():
    # Display content on the single page
    st.write("# Inventory Tracking Automation")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select an option:", ["Inventory Tracking"])
    
    # Display the selected functionality
    if page == "Inventory Tracking":
        
        
        show_compare()  # Use the imported function for comparison
  
if __name__ == "__main__":
    main()
