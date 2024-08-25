
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
        
                
                
            

                    
                    