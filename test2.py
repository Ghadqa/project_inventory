
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
import streamlit as st 
import pandas as pd
import openai
import logging
import datetime

import numpy as np
def show_compare():
    st.subheader("Upload files for comparison")

    # Define file paths
    file1 = "/home/poste1/inventory_compare/inventory23.csv"
    file2 = "/home/poste1/inventory_compare/Inventory_Board_20240821_103808 (copy).csv"

    if file1 is not None and file2 is not None:
        # Read the files into DataFrames
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # Vérification des colonnes requises
        required_columns = ["NEName", "Board Name", "Date Of Manufacture"]
        missing_columns = [col for col in required_columns if col not in df1.columns or col not in df2.columns]
     
        if missing_columns:
            st.error(f"The following columns are missing in the file(s): {', '.join(missing_columns)}")
            return
        #Count boards based on 'PN(BOM Code/Item)' and include 'Board Name'
        
        count_df1 = df1.groupby(['PN(BOM Code/Item)']).size().reset_index(name='Count_File1')
        count_df2 = df2.groupby(['PN(BOM Code/Item)']).size().reset_index(name='Count_File2')
        # Find added and removed boards
        added_boards = df2[~df2['NEName'].isin(df1['NEName'])]
        st.write("### Boards Added (Present in File 2 but not in File 1)")
        st.dataframe(added_boards)
        removed_boards = df1[~df1['NEName'].isin(df2['NEName'])]
        count_add = added_boards.groupby(['PN(BOM Code/Item)']).size().reset_index(name='new site')
        count_rmv = removed_boards.groupby(['PN(BOM Code/Item)']).size().reset_index(name='refresh')
        st.write("### Boards Removed (Present in File 1 but not in File 2)")
        st.dataframe(removed_boards)
        # Merge counts from both files
        merged_counts = pd.merge(count_df1, count_df2, on=['PN(BOM Code/Item)'], how='outer').fillna(0)
        
        # Calculate the total count
        merged_counts['Total_Count'] = merged_counts['Count_File1'] + merged_counts['Count_File2']
        
        merged_counts1= pd.merge(count_add, count_rmv, on=['PN(BOM Code/Item)'], how='outer').fillna(0)
        combined_df = pd.concat([merged_counts.set_index('PN(BOM Code/Item)'), 
                                 merged_counts1.set_index('PN(BOM Code/Item)')], axis=1, join='outer').reset_index()

        
        
        board_details_df1 = df1[['PN(BOM Code/Item)', 'Board Name', 'Date Of Manufacture']].drop_duplicates()
        board_details_df2 = df2[['PN(BOM Code/Item)', 'Board Name', 'Date Of Manufacture']].drop_duplicates()

        # Combine details from both files, prioritizing the most recent details
        board_details = pd.concat([board_details_df1, board_details_df2]).drop_duplicates(subset=['PN(BOM Code/Item)'], keep='last')

        # Merge board details with the combined DataFrame
        combined_df = pd.merge(combined_df, board_details, on='PN(BOM Code/Item)', how='left')
        
        # Add new column based on conditions
        combined_df['Status'] = np.where(
            (combined_df['new site'] > 0) & (combined_df['refresh'] == 0), 'Added Only',
            np.where(
                (combined_df['new site'] == 0) & (combined_df['refresh'] > 0), 'Removed Only',
                'Both'
            )
        )
        # Add a numeric column based on 'Status'
        combined_df['Status_Numeric'] = combined_df['Status'].map({
            'Added Only': 1,
            'Removed Only': 1,
            'Both': 2
        })
        boards_data = {
                "FModule": "",
                "GTMU": "Transmission processing 2G GSM (No Bandwidth)",
                "FAN": "cards are found at all the sites(No Bandwidth)",
                "UPEU": "Power module",
                "UEIU": "(No Bandwidth)",
                "UMPT": "Unité Principale de Traitement et de Transmission Universelle: dans BUU5900 se trouve dans le slot 7",
                "WMPT": "Transmission processing 3G WCDMA",
                "LMPT": "Transmission processing 4G LTE",
                "UBBP": "Process uplink and downlink baseband signals",
                "WBBP": "Processes uplink and downlink baseband signals 3G",
                "LBBP": "Processes uplink and downlink baseband signals 4G",
                "MRRU": "The frequency bands can vary depending on network specifications (800 MHz, 900 MHz, 1800 MHz, 2100 MHz)",
                "RFU": "The frequency bands can vary depending on network specifications (800 MHz, 900 MHz, 1800 MHz, 2100 MHz)",
                "AUU": "The frequency bands can vary depending on network specifications (700 MHz, 1800 MHz, 2100 MHz)",
                "PSU": "Used for power supply (No Bandwidth)",
                "PDB": "Used for power supply (No Bandwidth)",
                "PMI": "Used for power supply (No Bandwidth)",
                "TCU": "Used for control and synchronization (No Bandwidth)",

               
            }
        combined_df['Description'] = combined_df['Board Name'].map(boards_data)
        st.write("### boards count")
        st.dataframe(combined_df)
        excel_file_path = "/home/poste1/inventory_compare/combined_data.xlsx"
        combined_df.to_excel(excel_file_path, index=False, engine='openpyxl')
        st.success(f"Combined data saved to {excel_file_path}")

        # Prepare data for histogram
        histogram_data = combined_df[['PN(BOM Code/Item)', 'Board Name', 'new site', 'refresh']].melt(
            id_vars=['PN(BOM Code/Item)', 'Board Name'], var_name='Category', value_name='Count')

        # Create a new column that combines 'PN(BOM Code/Item)' and 'Board Name'
        histogram_data['Identifier'] = histogram_data['PN(BOM Code/Item)'] + " - " + histogram_data['Board Name']

        # Create the histogram-like bar chart
        fig_histogram = px.bar(
            histogram_data,
            x='Identifier',
            y='Count',
            color='Category',
            title="New Site vs Refresh Boards",
            labels={"Identifier": "PN(BOM Code/Item) - Board Name", "Count": "Number of Boards"},
            color_discrete_map={'new site': 'royalblue', 'refresh': 'tomato'},
            text='Count'
        )

        # Customize layout
        fig_histogram.update_layout(
            xaxis_title="PN(BOM Code/Item) - Board Name",
            yaxis_title="Number of Boards",
            xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
            barmode='stack',     # Stack bars to create a histogram-like appearance
            showlegend=True
        )

        # Add value labels on the bars
        fig_histogram.update_traces(texttemplate='%{text}', textposition='outside', cliponaxis=False)

        # Display the chart
        st.plotly_chart(fig_histogram)

    

        # Dropdown to select PN(BOM Code/Item)
        pn_options = combined_df['PN(BOM Code/Item)'].unique()
        selected_pn = st.sidebar.selectbox("Select PN(BOM Code/Item)", options=pn_options)

        # Filter DataFrame based on selected PN(BOM Code/Item)
        filtered_by_pn = combined_df[combined_df['PN(BOM Code/Item)'] == selected_pn]

        # Display filtered DataFrame
        st.write(f"### Details for PN(BOM Code/Item) {selected_pn}")
        st.dataframe(filtered_by_pn)

    
        df1['Date Of Manufacture'] = pd.to_datetime(df1['Date Of Manufacture'], errors='coerce')
        df2['Date Of Manufacture'] = pd.to_datetime(df2['Date Of Manufacture'], errors='coerce')

        # Replace NaT with a default date or handle them as needed
        df1['Date Of Manufacture'].fillna(pd.Timestamp.min, inplace=True)
        df2['Date Of Manufacture'].fillna(pd.Timestamp.min, inplace=True)

          # Add date filtering options
        st.sidebar.subheader("Filter by Year")
        combined_df['Date Of Manufacture'] = pd.to_datetime(combined_df['Date Of Manufacture'], errors='coerce')
        # Extract year from 'Date Of Manufacture'
        combined_df['Year'] = combined_df['Date Of Manufacture'].dt.year

        selected_year = st.sidebar.selectbox("Select a year", options=sorted(combined_df['Year'].unique()), index=0)

        # Filter by selected year
        filtered_by_year = combined_df[combined_df['Year'] == selected_year]

        # Display the filtered boards
        st.write(f"### Boards for Year {selected_year} ({len(filtered_by_year)})")
        st.dataframe(filtered_by_year)
      

        monthly_data = filtered_by_year[['PN(BOM Code/Item)', 'Board Name', 'Date Of Manufacture', 'new site', 'refresh']].melt(
        id_vars=['PN(BOM Code/Item)', 'Board Name', 'Date Of Manufacture'], var_name='Category', value_name='Count')

        # Create a combined identifier for better labeling
        monthly_data['Identifier'] = monthly_data['PN(BOM Code/Item)'] + " - " + monthly_data['Board Name']
    
         
        fig_monthly = px.bar(
            monthly_data,
            x='Identifier',
            y='Count',
            color='Category',
            title=f"Distribution of New Site and Refresh Boards on year {selected_year}",
            labels={"Month": "Month", "Count": "Number of Boards"},
            color_discrete_map={'new site': 'royalblue', 'refresh': 'tomato'},
            text='Count',
            barmode='group'
        )

        fig_monthly.update_layout(
            xaxis_title="Date Of Manufacture",
            yaxis_title="Number of Boards",
            xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
            showlegend=True
        )
        fig_monthly.update_traces(texttemplate='%{text}', textposition='outside', cliponaxis=False)
        st.plotly_chart(fig_monthly)
        # Ensure the 'Date Of Manufacture' column is in datetime format
        filtered_by_year['Date Of Manufacture'] = pd.to_datetime(filtered_by_year['Date Of Manufacture'], errors='coerce')

        # Extract month from 'Date Of Manufacture'
        filtered_by_year['Month'] = filtered_by_year['Date Of Manufacture'].dt.month

        # Dropdown to select month
        selected_month = st.sidebar.selectbox("Select Month", options=range(1, 13), format_func=lambda x: pd.to_datetime(x, format='%m').strftime('%B'))

        # Filter DataFrame based on the selected month
        filtered_df = filtered_by_year[filtered_by_year['Month'] == selected_month]


       
        # Bar Chart of New Site vs Refresh Boards by Month
        monthly_data = filtered_df[['PN(BOM Code/Item)', 'Board Name', 'Date Of Manufacture', 'new site', 'refresh']].melt(
            id_vars=['PN(BOM Code/Item)', 'Board Name', 'Date Of Manufacture'], var_name='Category', value_name='Count')

        # Create a combined identifier for better labeling
        monthly_data['Identifier'] = monthly_data['PN(BOM Code/Item)'] + " - " + monthly_data['Board Name']

        fig_monthly = px.bar(
            monthly_data,
            x='Identifier',
            y='Count',
            color='Category',
            title=f"Monthly{selected_month} on year {selected_year} Distribution of New Site and Refresh Boards",
            labels={"Identifier": "PN(BOM Code/Item) - Board Name", "Count": "Number of Boards"},
            color_discrete_map={'new site': 'royalblue', 'refresh': 'tomato'},
            text='Count',
            barmode='group'
        )

        fig_monthly.update_layout(
            xaxis_title="PN(BOM Code/Item) - Board Name",
            yaxis_title="Number of Boards",
            xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
            showlegend=True
        )

        fig_monthly.update_traces(texttemplate='%{text}', textposition='outside', cliponaxis=False)

        st.plotly_chart(fig_monthly)
        

        filtered_both = filtered_by_year[filtered_by_year['PN(BOM Code/Item)'] == selected_pn]
        st.write(f"### Board with PN(BOM Code/Item) ={selected_pn}for Year {selected_year})")
        st.write(filtered_both)
        
                
                
            

                    


def chat():
    st.title("Upload Two Files for Comparison and Ask Questions")

    # File upload section for two files
    uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) == 2:
        try:
            # Read both files into dataframes
            df1 = pd.read_csv(uploaded_files[0])
            df2 = pd.read_csv(uploaded_files[1])

            st.write(f"Contents of {uploaded_files[0].name}:")
            st.write(df1)
            st.write(f"Contents of {uploaded_files[1].name}:")
            st.write(df2)

            # Compare the two dataframes and find common and different values
            common_df, diff_file1_df, diff_file2_df = compare_nename_columns(df1, df2, "NEName")

            if common_df is not None:
                st.write("### Common NEName Values:")
                st.write(common_df)

                st.write("### NEName Values in File 1 but not in File 2:")
                st.write(diff_file1_df)

                st.write("### NEName Values in File 2 but not in File 1:")
                st.write(diff_file2_df)

                # Set your Azure OpenAI key and endpoint
                openai.api_type = "azure"
                openai.api_key = "25117d14b1574833b0995c5c5a873ff5"
                openai.api_base = "https://nice.openai.azure.com/"
                openai.api_version = "2023-05-15"

                question = st.text_input("Ask a question related to the comparison")

                if question:
                    try:
                        answer = ask_openai_with_chunks(question, diff_file1_df, diff_file2_df,common_df, chunk_size=100)
                        st.write(f"Answer: {answer}")
                        logging.info(f"Question: {question} | Answer: {answer}")
                    except Exception as e:
                        st.error("There was an issue with processing your question.")
                        logging.error(f"Error processing question: {question} | Exception: {str(e)}")
        except Exception as e:
            st.error("There was an error processing the files.")
            logging.error(f"Error processing files: {str(e)}")
    else:
        st.warning("Please upload exactly two CSV files.")
        logging.warning("User did not upload exactly two CSV files.")


def filter_question(question, df):
    # Simple example to detect if a question is related to the data
    keywords = ["boards", "inventory", "date"]
    if any(keyword in question.lower() for keyword in keywords):
        return True
    return False
def ask_openai_with_chunks(question: str, df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame, chunk_size: int) -> str:
    """
    Queries Azure OpenAI with data chunks and handles specific questions about added/recovered boards.

    Args:
    - question (str): The question to ask.
    - df1 (pd.DataFrame): The DataFrame containing data for added boards.
    - df2 (pd.DataFrame): The DataFrame containing data for recovered boards.
    - chunk_size (int): The number of rows per chunk.

    Returns:
    - str: The combined answer from all chunks.
    """

    if "new boards with more recent manufacture dates" in question.lower():
        post_dates_df = compare_dates_later(df1, df2, "Date_Of_Manufacture")
        
        if not post_dates_df.empty:
            boards_info = [
                f"NEName: {row['NEName']}, Board: {row['Board Name']}, SN: {row['s']}, Date: {row['Date_Of_Manufacture'].strftime('%Y-%m-%d')}"
                for _, row in post_dates_df.iterrows()
            ]
            return f"The new boards with more recent manufacture dates in File 2 are:\n" + "\n".join(boards_info)
        else:
            return "No new boards with more recent manufacture dates found in File 2."
    
    if "boards on date" in question.lower():
        df1['Date_Of_Manufacture'] = pd.to_datetime(df1['Date_Of_Manufacture'], errors='coerce').fillna(pd.Timestamp.min)
        df2['Date_Of_Manufacture'] = pd.to_datetime(df2['Date_Of_Manufacture'], errors='coerce').fillna(pd.Timestamp.min)
        
        filter_option = st.sidebar.selectbox("Choose date filter type", ["Single Date", "Date Range"])

        if filter_option == "Single Date":
            selected_date = st.sidebar.date_input("Select a date", min_value=datetime(2000, 1, 1), max_value=datetime(2100, 12, 31))
            filtered_df1 = df1[df1['Date_Of_Manufacture'] == pd.to_datetime(selected_date)]
            filtered_df2 = df2[df2['Date_Of_Manufacture'] == pd.to_datetime(selected_date)]
            
            if not filtered_df1.empty or not filtered_df2.empty:
                boards_info = [
                    f"NEName: {row['NEName']}, Board: {row['Board Name']}, SN: {row['sn_bar_code']}, Date: {row['Date_Of_Manufacture'].strftime('%Y-%m-%d')}"
                    for _, row in pd.concat([filtered_df1, filtered_df2]).iterrows()
                ]
                return f"The boards manufactured on {selected_date.strftime('%Y-%m-%d')} are:\n" + "\n".join(boards_info)
            else:
                return f"No boards found on {selected_date.strftime('%Y-%m-%d')}."
        
        elif filter_option == "Date Range":
            start_date = st.sidebar.date_input("Select start date", min_value=datetime(2000, 1, 1), max_value=datetime(2100, 12, 31))
            end_date = st.sidebar.date_input("Select end date", min_value=datetime(2000, 1, 1), max_value=datetime(2100, 12, 31))
            filtered_df1 = df1[(df1['Date_Of_Manufacture'] >= pd.to_datetime(start_date)) & (df1['Date_Of_Manufacture'] <= pd.to_datetime(end_date))]
            filtered_df2 = df2[(df2['Date_Of_Manufacture'] >= pd.to_datetime(start_date)) & (df2['Date_Of_Manufacture'] <= pd.to_datetime(end_date))]

            if not filtered_df1.empty or not filtered_df2.empty:
                boards_info = [
                    f"NEName: {row['NEName']}, Board: {row['Board_Name']}, SN: {row['sn_bar_code']}, Date: {row['Date_Of_Manufacture'].strftime('%Y-%m-%d')}"
                    for _, row in pd.concat([filtered_df1, filtered_df2]).iterrows()
                ]
                return f"The boards manufactured between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')} are:\n" + "\n".join(boards_info)
            else:
                return f"No boards found between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}."

    elif "how many boards added" in question.lower():
        return f"There are {len(df2)} boards added."
    
    elif "which boards added" in question.lower():
        added_boards = df2[["NEName", "Board Name"]]
        st.write("### Added Boards Information")
        st.dataframe(added_boards)
        return ""
        
    elif "how many boards recovered" in question.lower():
        return f"There are {len(df1)} boards to be recovered."
    
    elif "which boards recovered" in question.lower():
        recovered_boards = df1[["NEName","PN (BOM Code/Item)","Board Name"]]
        st.write("### Recovered Boards Information")
        st.dataframe(recovered_boards)
        return ""
    
    elif "give me information about new sites" in question.lower():
        new_sites_info = df2[["NEName", "Board_Name", "Date_Of_Manufacture"]]
        return new_sites_info.to_string(index=False)
    
    elif "how many boards new site nename=" in question.lower():
        nename_value = question.split('=')[-1].strip().upper()
        filtered_df = df2[df2['NEName'] == nename_value]
        

        if not filtered_df.empty:
            board_count = len (filtered_df['Board_Name'])
            st.write(f"There are {board_count} boards in NEName = {nename_value}")
            st.dataframe(filtered_df[["Board_Name", "Board Type"]])
        else:
            st.write(f"No boards found for NEName = {nename_value}")

    elif "new site " in question.lower(): 
        new_site = set(df2["NEName"])
        return ', '.join(new_site)

   
   
    
    chunks = chunk_dataframe(df1, chunk_size)
    answers = []
    for chunk in chunks:
        context = chunk.head(5).to_dict()  # Convert a portion to avoid excessive context
        prompt = f"Based on the following data: {context}, {question}"
        
        try:
            response = openai.Completion.create(
                engine="inventory_gpt",
                prompt=prompt,
                max_tokens=150
            )
            answers.append(response.choices[0].text.strip())
        except Exception as e:
            answers.append(f"Error: {str(e)}")

    return "\n".join(answers)

def compare_nename_columns(df1: pd.DataFrame, df2: pd.DataFrame, column_name: str) -> tuple:
    """
    Compare NEName columns of two dataframes and find common and different values.

    Args:
    - df1 (pd.DataFrame): The first dataframe.
    - df2 (pd.DataFrame): The second dataframe.
    - column_name (str): The column to compare.

    Returns:
    - tuple: A tuple containing dataframes with common values, values in df1 not in df2, and values in df2 not in df1.
    """
    common = df1[df1[column_name].isin(df2[column_name])]
    diff_file1 = df1[~df1[column_name].isin(df2[column_name])]
    diff_file2 = df2[~df2[column_name].isin(df1[column_name])]
    return common, diff_file1, diff_file2

def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> list:
    """
    Splits a DataFrame into smaller chunks.

    Args:
    - df (pd.DataFrame): The DataFrame to split.
    - chunk_size (int): The number of rows per chunk.

    Returns:
    - list: A list of DataFrame chunks.
    """
    return [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]

def get_recovered_boards(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Identify boards that are present in the old inventory but missing from the new inventory.

    Args:
    - df1 (pd.DataFrame): The old inventory dataframe.
    - df2 (pd.DataFrame): The new inventory dataframe.

    Returns:
    - pd.DataFrame: A dataframe containing boards to be recovered.
    """
    # Find boards in the old inventory that are not in the new inventory
    recovered_boards = df1[~df1["NEName"].isin(df2["NEName"])]

    return recovered_boards

def convert_to_datetime(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name])
    return df
def compare_dates_later(db_df, new_df, column_name):
    # Convertir les colonnes de dates en format datetime, si ce n'est déjà fait
    db_df[column_name] = pd.to_datetime(db_df[column_name], errors='coerce')
    new_df[column_name] = pd.to_datetime(new_df[column_name], errors='coerce')

    # Filtrer les entrées de new_df qui ont une date postérieure à celle de db_df
    post_dates = new_df[new_df[column_name] > db_df[column_name].max()]
    
    return post_dates
