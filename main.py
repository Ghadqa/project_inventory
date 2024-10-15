import streamlit as st
from functions import show_compare, dashboard  # Assuming these functions are already defined

# Page configuration
st.set_page_config(page_title="Huawei Inventory Tracking Automation", page_icon="🔧", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;  /* Light background for contrast */
            font-family: Arial, sans-serif;
        }
        .stButton button {
            background-color: #C0392B; /* Darker red */
            color: white;
            font-size: 18px;
            margin: 20px;  /* Margin to separate buttons */
            border-radius: 5px;
            padding: 15px 20px;  /* Adjusted padding for better button size */
            border: none;
            transition: background-color 0.3s;
        }
        .stButton button:hover {
            background-color: #A93226; /* Lighter red on hover */
        }
        h1 {
            color: #C0392B; /* Red color for headers */
            text-align: center;
            margin-top: 30px;
        }
        h2, h3 {
            color: #2C3E50; /* Dark color for subheaders */
            text-align: center;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #C0392B; /* Dark red for footer */
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }
        .info-box {
            background-color: white;
            border: 2px solid #C0392B; /* Red border for info box */
            padding: 20px;
            border-radius: 10px;
            margin-top: 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Shadow for depth */
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'  # Set the default page as 'Home'

# Function to handle navigation
def go_to_page(page):
    st.session_state['page'] = page

# Render the page based on the session state
if st.session_state['page'] == 'Home':
    st.title("Welcome to Huawei Inventory Tracking Automation")
    st.subheader("Automating the tracking of network boards for performance and efficiency")
    
    # Home buttons on the same line, positioned on each side of the page
    col1, col2 = st.columns([1, 1])  # Create two columns of equal width

    with col1:
        if st.button('Start Board Tracking'):
            go_to_page('Board Tracking')

    with col2:
        
        if st.button('Explore Inventory'):
            go_to_page('Visualizations')

elif st.session_state['page'] == 'Board Tracking':
    st.title("Board Tracking")
 
    show_compare()  # Placeholder for your board tracking function
    
    # Back button to go to Home
    if st.button('Back to Home'):
        go_to_page('Home')

elif st.session_state['page'] == 'Visualizations':
    #st.title("Inventory Visualizations")
   
    dashboard()  # Placeholder for your visualization function
    
    # Back button to go to Home
    if st.button('Back to Home'):
        go_to_page('Home')

# Footer
st.markdown('<div class="footer">For more information, contact the technical support team | Huawei</div>', unsafe_allow_html=True)
