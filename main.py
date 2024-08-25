 
import pandas as pd
import streamlit as st
import plotly.express as px
import sqlite3
import pdfkit
from jinja2 import Template
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

from test2 import chat , show_compare
st.set_page_config(page_title="Inventory Dashboard", page_icon=":bar_chart:", layout="wide")
def main():
    # Barre latérale pour la navigation
     
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["compare data","visualize data", "ask chat"])

   
    if page =="compare data": 
        show_compare()
   
    elif page=="ask chat":
         chat()
        
if __name__ == "__main__":
    main()
 