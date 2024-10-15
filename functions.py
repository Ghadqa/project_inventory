from io import StringIO
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import re
import pandas as pd
import streamlit as st
import plotly.express as px
from io import StringIO
from jinja2 import Template
from jinja2 import Template  # Import Template from jinja2
import pdfkit



def extract_band_info(manufacture_data):
    """
    Extract TX and RX frequency ranges from manufacture data and return them as numeric intervals.
    """
    match = re.search(r'TX(\d+)-(\d+)MHz/RX(\d+)-(\d+)MHz', manufacture_data)
    if match:
        tx_range = (int(match.group(1)), int(match.group(2)))
        rx_range = (int(match.group(3)), int(match.group(4)))
        return tx_range, rx_range
    return None, None


# Load the CSV file into a DataFrame
def filter_and_visualize_data():
    df = pd.read_csv("/home/poste1/inventory_compare/_Inventory_Board_2023.csv")

    # Add the 'Band Unit' column based on 'Manufacturer Data'
    df['Band Unit'] = df['Manufacturer Data'].apply(lambda x: extract_band_info(x)[0])

    # Check if the DataFrame is empty
    if df.empty:
        st.write("The DataFrame is empty or could not be processed.")
        return

    # Include custom CSS for styling
    st.markdown(
        """
        <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f8f9fa;  /* Light background color */
        }
        h1 {
            color: #007bff;  /* Title color */
        }
        h2 {
            color: #343a40;  /* Subtitle color */
        }
        .stSelectbox {
            margin-bottom: 20px;  /* Margin for dropdown */
        }
        table {
            width: 100%;  /* Table width */
            border-collapse: collapse;  /* Remove space between borders */
        }
        table, th, td {
            border: 1px solid #007bff;  /* Border color */
        }
        th, td {
            padding: 8px;  /* Cell padding */
            text-align: left;  /* Left align text */
        }
        th {
            background-color: #007bff;  /* Header background color */
            color: white;  /* Header text color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title of the application
    st.title("Inventory Board Data")

    # Filter data based on the board name
    board = st.sidebar.selectbox("Select the board", df['Board Name'].unique())

    # Filter the DataFrame based on the selected board
    filtered_by_board = df[df["Board Name"] == board]

    # Display the filtered information
    st.subheader(f"Networks information for the selected board: {board}")
    st.dataframe(filtered_by_board[['NEType', 'NEFdn', 'NEName', 'Board Name', 'Board Type', 'PN(BOM Code/Item)', 'Manufacturer Data', 'Band Unit']])

    # Visualize the distribution of NEName by Date of Manufacture
    st.subheader("Distribution of NEName by Date of Manufacture")
    df["Date Of Manufacture"] = pd.to_datetime(df["Date Of Manufacture"], errors='coerce')
    df['Year'] = df["Date Of Manufacture"].dt.year

    # Grouping the data by NEName and Year
    distribution_data = df.groupby(['Year', 'NEName']).size().reset_index(name='Count')

    # Create a bar chart
    fig = px.bar(distribution_data, x='Year', y='Count', color='NEName',
                 title="Distribution of NEName by Year of Manufacture",
                 labels={'Count': 'Number of Boards', 'Year': 'Year'},
                 barmode='group',
                 height=400)

    st.plotly_chart(fig)

    # Filter data by year of manufacture
    selected_year = st.sidebar.number_input("Enter the year of manufacture", min_value=2000, max_value=2030, value=2023)

    filtered_by_year = df[df["Year"] == selected_year]
    st.write(f"Boards information for the year {selected_year}:")
    st.write(filtered_by_year)

    # Filter by NEName
    nename = st.selectbox("Select the NEName", df['NEName'].unique())
    filtered_by_nename = df[df["NEName"] == nename]
    st.write(f"Boards information for selected NEName {nename}:")
    st.write(filtered_by_nename)

    if not filtered_by_nename.empty:
        board_name = st.selectbox(f"Select the board name for {nename}", filtered_by_nename['Board Name'].unique())
        df_filtered = filtered_by_nename[filtered_by_nename["Board Name"] == board_name]

        # Pre-defined dictionary for boards' description
        boards_data = {
            "FModule": "",
            "GTMU": "Transmission processing 2G GSM (No Bandwidth)",
            "FAN": "Cards are found at all the sites (No Bandwidth)",
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
            "TCU": "Used for control and synchronization (No Bandwidth)"
        }

        if board_name in boards_data:
            st.write(f"Information for Board Name {board_name}: {boards_data[board_name]}")

        if not df_filtered.empty:
            st.write(f"Manufacturing details for {board_name}:")
            st.write(df_filtered[['Board Name', 'PN(BOM Code/Item)', 'Date Of Manufacture']])
        else:
            st.write("No data available for the selected board.")
    else:
        st.write("No valid data for the selected NEName or Board Name.")
# Visualize data
def visualize_data(filtered_by_date, df):
    # Bar Chart: Count of Board Types for filtered data
    if not filtered_by_date.empty:
        board_type_counts = filtered_by_date['Board Name'].value_counts().reset_index()
        board_type_counts.columns = ['Board Name', 'Count']
        fig_bar = px.bar(board_type_counts, x='Board Name', y='Count', title='Count of Board Names in Filtered Data')
        st.plotly_chart(fig_bar)
    else:
        st.write("No data available for the selected filters.")

    # Pie Chart: Distribution of Board Types in the overall dataset
    board_type_counts_overall = df['Board Name'].value_counts().reset_index()
    board_type_counts_overall.columns = ['Board Name', 'Count']
    fig_pie = px.pie(board_type_counts_overall, values='Count', names='Board Name', title='Board Names Distribution')
    st.plotly_chart(fig_pie)


def read_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        excel_df = pd.read_excel(file)
        csv_data = StringIO()
        excel_df.to_csv(csv_data, index=False)
        csv_data.seek(0)
        return pd.read_csv(csv_data)

def generate_report(classified_data, filename="final_report.pdf"):
    # Create an HTML template for the report
    html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Huawei Board Classification Report</title>
    <style>
        table { width: 100%; border-collapse: collapse; }
        table, th, td { border: 1px solid black; }
        th, td { padding: 8px; text-align: left; }
    </style>
</head>
<body>
    <h1>Huawei Board Classification Report</h1>
    <h2>Project Overview</h2>
    <p>This report summarizes the classification of network boards after installation at specific sites.</p>
    <p>As a telecommunications solutions provider, Huawei is responsible for deploying network equipment, such as essential electronic cards, ensuring optimal performance of installed infrastructures, including 4G networks.</p>
    <h2>Classification Summary</h2>
    <table>
        <thead>
            <tr>
                <th>Site Name</th>
                <th>Board Name</th>
                <th>Date Of Manufacture</th>
                <th>SN Bar Code</th>
                <th>Category</th>
            </tr>
        </thead>
        <tbody>
            {% for row in data %}
            <tr>
                <td>{{ row['NEName'] }}</td>
                <td>{{ row['Board Name'] }}</td>
                <td>{{ row['Date Of Manufacture'] }}</td>
                <td>{{ row['SN Bar Code'] }}</td>
                <td>{{ row['Category'] }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    <h2>Conclusion</h2>
    <p>The automation of board tracking is crucial for maintaining accurate inventory records and ensuring the efficient operation of network infrastructures. This report will aid in further discussions and decision-making processes with clients.</p>
</body>
</html>
"""

    template = Template(html_template)
    html_report = template.render(data=classified_data.to_dict(orient='records'))

    # Convert HTML to PDF
    pdfkit.from_string(html_report, filename)

def show_compare():
        #
    uploaded_file1 = st.file_uploader("Upload old inventory export", type=["csv", "xlsx"])
    uploaded_file2 = st.file_uploader("Upload new inventory export", type=["csv", "xlsx"])

     

    if uploaded_file1 is not None and uploaded_file2 is not None:
          
        df1 = read_file(uploaded_file1)
        df2 = read_file(uploaded_file2)

        required_columns = ["NEName", "Board Name", "Date Of Manufacture", "PN(BOM Code/Item)", "Manufacturer Data"]
        missing_columns = [col for col in required_columns if col not in df1.columns or col not in df2.columns]

        if missing_columns:
            st.error(f"The following columns are missing in the file(s): {', '.join(missing_columns)}")
            return

        boards_data = {
            "FModule": "Module Description",
            "GTMU": "Transmission processing 2G GSM (No Bandwidth)",
            "FAN": "Cards are found at all the sites (No Bandwidth)",
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
            "TCU": "Used for control and synchronization (No Bandwidth)"
        }
        col1, col2 = st.columns(2)
        with col1:
   
            newsite = df2[~df2['NEName'].isin(df1['NEName'])]
            st.write("### New Site")
            st.write(newsite)

    
            rru_info = newsite[newsite['Board Name'].str.contains('RRU', na=False)]
            non_rru_info = newsite[~newsite['Board Name'].str.contains('RRU', na=False)]

        with col2:
                    # Add sidebar header with red tex

            # Add custom label with red text for the selectbox
            st.sidebar.markdown(
                "<p style='color: #FF0000;'>Select a NEName to display details for the new site:</p>",  # Red text for the selectbox label
                unsafe_allow_html=True
            )
            # Display a styled selectbox for NEName with unique options
            unique_nenames = newsite['NEName'].unique()

            # Selectbox to choose the NEName
            selected_name = st.sidebar.selectbox(
                "",
                unique_nenames
            )

            
            if selected_name:
                st.write(f"### Details for NEName: {selected_name}")


                combined_info = pd.concat([
                    rru_info[rru_info['NEName'] == selected_name].assign(Type='RRU'),
                    non_rru_info[non_rru_info['NEName'] == selected_name].assign(Type='Non-RRU')
                ])

                if not combined_info.empty:
                
                    total_boards = combined_info.shape[0]
                    st.write(f"**Total Number of Boards:** {total_boards}")

                    combined_counts = combined_info.groupby(['Board Name', 'PN(BOM Code/Item)']).size().reset_index(name='Count')

            
                    fig_combined = px.bar(combined_counts, x='Board Name', y='Count', color='PN(BOM Code/Item)',
                                        title='Number of Entries by Board Name and PN Code',
                                        labels={'Board Name': 'Board Name', 'Count': 'Count', 'PN(BOM Code/Item)': 'PN Code'})
                    st.plotly_chart(fig_combined)
                else:
                    st.write("No information available for this NEName.")
                


            #unique_nenames = newsite['NEName'].unique()
            #selected_name = st.sidebar.selectbox("Select NEName to display details for new site ", unique_nenames)
        new_boards = df2[~df2['PN(BOM Code/Item)'].isin(df1['PN(BOM Code/Item)'])]

    



        new_boards_newsite = new_boards[new_boards['NEName'].isin(newsite['NEName'])]

   
        dismantled_boards = df2[df2['PN(BOM Code/Item)'].isin(df1['PN(BOM Code/Item)'])]
        dismantled_boards_info = dismantled_boards[['Board Name', 'PN(BOM Code/Item)', 'Date Of Manufacture', 'Manufacturer Data']].drop_duplicates()
        dismantled_boards_info['Band_Info'] = dismantled_boards_info['Manufacturer Data'].apply(lambda x: extract_band_info(x)[0] if x else None)
            # Step 2: Create tuples of (Board Name, PN(BOM Code/Item))
        dismantled_boards['Board_PN_Tuple'] = dismantled_boards.apply(
            lambda row: f"{row['Board Name']} ({row['PN(BOM Code/Item)']})", axis=1
        )

        # Step 3: Count occurrences of each tuple
        tuple_counts = dismantled_boards['Board_PN_Tuple'].value_counts().reset_index()
        tuple_counts.columns = ['Board_PN_Tuple', 'Count']

        # Step 3: Clean the 'Date Of Manufacture' column and extract Year
        dismantled_boards_info['Date Of Manufacture'] = dismantled_boards_info['Date Of Manufacture'].replace(r'^\s*$', pd.NaT, regex=True)  # Replace empty strings with NaT
        dismantled_boards_info['Year Of Manufacture'] = pd.to_datetime(dismantled_boards_info['Date Of Manufacture'], errors='coerce').dt.year

        # Step 4: Count occurrences by Year of Manufacture
        year_counts = dismantled_boards_info['Year Of Manufacture'].value_counts().reset_index()
        year_counts.columns = ['Year Of Manufacture', 'Count']
        col1, col2,col3 = st.columns(3)
        with col1:
            st.write("### Dismantled Items Information")
            st.write(dismantled_boards_info)
        with col2:
                # Visualization of dismantled boards by tuple
            if not tuple_counts.empty:
                st.write("### Dismantled Boards Count by (Board Name, PN(BOM Code/Item)) ")
                
                # Create a bar chart for the counts of Board Name and PN BOM Code pairs
                fig_dismantled_tuples = px.bar(
                    tuple_counts,
                    x='Board_PN_Tuple',
                    y='Count',
                    title='Count of Dismantled Boards by (Board Name, PN(BOM Code/Item)) Tuples',
                    labels={'Count': 'Number of Dismantled Boards'},
                    color='Count',
                    text='Count'  # Show counts on bars
                )
                
                # Update the x-axis to improve readability
                fig_dismantled_tuples.update_layout(
                    xaxis_title='(Board Name, PN(BOM Code/Item))',
                    xaxis_tickangle=-45
                )

                # Show the plot
                st.plotly_chart(fig_dismantled_tuples)
            else:
                st.write("No dismantled boards found for the selected sites.")
        with col3 : 
                    # Visualization of dismantled boards by Year of Manufacture
            if not year_counts.empty:
                st.write("### Dismantled Boards Count by Year of Manufacture")
                
                # Create a bar chart for the counts of boards by year
                fig_year_counts = px.bar(
                    year_counts,
                    x='Year Of Manufacture',
                    y='Count',
                    title='Count of Dismantled Boards by Year of Manufacture',
                    labels={'Count': 'Number of Dismantled Boards'},
                    color='Count',
                    text='Count'  # Show counts on bars
                )
                
                # Update the x-axis for better readability
                fig_year_counts.update_layout(
                    xaxis_title='Year of Manufacture',
                    xaxis_tickangle=-45
                )

                # Show the plot
                st.plotly_chart(fig_year_counts)
            else:
                st.write("No dismantled boards found for the selected sites.")
        newsite['Band_Info'] = newsite['Manufacturer Data'].apply(lambda x: extract_band_info(x)[0] if x else None)

        newsite['Category'] = np.nan

        #---------new items in neww site= new ------#
        newsite.loc[newsite['NEName'].isin(new_boards_newsite['NEName']), 'Category'] = 'new'


        rru_boards = newsite[newsite['Board Name'].str.startswith(('RRU', 'MRRU', 'LRRU'))]

        def update_category_for_board(row, matching_rrus, category):
           
            tx_range_old, rx_range_old = extract_band_info(row['Manufacturer Data'])

          
            if tx_range_old is None or rx_range_old is None:
                return

            for _, new_row in matching_rrus.iterrows():
                tx_range_new, rx_range_new = extract_band_info(new_row['Manufacturer Data'])

                if tx_range_new == tx_range_old and rx_range_new == rx_range_old:
                    newsite.at[row.name, 'Category'] = category
                    break

      
        for _, row in rru_boards.iterrows():
           
            matching_rrus = dismantled_boards_info[
                (dismantled_boards_info['PN(BOM Code/Item)'] == row['PN(BOM Code/Item)']) &
                (dismantled_boards_info['Band_Info'] == extract_band_info(row['Manufacturer Data'])[0])
            ]
            if not matching_rrus.empty:
                update_category_for_board(row, matching_rrus, 'refresh')
            else:
                newsite.at[row.name, 'Category'] = 'extension'
        bbp_boards = newsite[newsite['Board Name'].str.startswith('BBP')]
        for _, row in bbp_boards.iterrows():
            # Extract band information for the current BBP board
            board_band_info = extract_band_info(row['Manufacturer Data'])[0]
            
            # Check if there's a matching dismantled WBBP or LBBP with the same PN and band info
            matching_bbp = dismantled_boards_info[
                (dismantled_boards_info['PN(BOM Code/Item)'] == row['PN(BOM Code/Item)']) &
                (dismantled_boards_info['Band_Info'] == board_band_info) &
                dismantled_boards_info['Board Name'].str.startswith(('WBBP', 'LBBP'))
            ]
            
            if not matching_bbp.empty:
                # If a matching dismantled WBBP/LBBP is found, categorize as 'refresh'
                newsite.at[row.name, 'Category'] = 'refresh'
            else:
                # If no match is found, categorize as 'extension'
                newsite.at[row.name, 'Category'] = 'extension'


        col1, col2 = st.columns(2)
        with col1:
            st.write("### Classification of Boards in the Network ")
            extrait_refresh=newsite[['NEName', 'PN(BOM Code/Item)','Board Name', 'Date Of Manufacture', 'Category']]
            #extrait_refresh['Description'] = extrait_refresh['Board Name'].map(boards_data)

            st.write(extrait_refresh)
        with col2: 
            st.write("### Visualization of Boards by Category Type")
        
            # Count the occurrences of each category
            category_counts = extrait_refresh['Category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            
            # Filter categories if needed (optional)
            # You can customize the following line to include only specific categories if desired
            category_counts = category_counts[category_counts['Category'].isin(['refresh', 'extension', 'new'])]

            # Create a bar chart for the counts of each category
            fig_category_counts = px.bar(
                category_counts,
                x='Category',
                y='Count',
                title='Count of Boards by Category Type (Refresh, Extension, New)',
                labels={'Count': 'Number of Boards'},
                color='Count',
                text='Count'  # Show counts on bars
            )
            
            # Show the plot
            st.plotly_chart(fig_category_counts)

            

        
        # Group by PN(BOM Code/Item) and Date Of Manufacture for 'new' category
        new_count = extrait_refresh[extrait_refresh['Category'] == 'new'].groupby(['PN(BOM Code/Item)', 'Date Of Manufacture','NEName']).size().reset_index(name='Count_New')

        # Group by PN(BOM Code/Item) and Date Of Manufacture for 'extension' category
        extension_count = extrait_refresh[extrait_refresh['Category'] == 'extension'].groupby(['PN(BOM Code/Item)', 'Date Of Manufacture','NEName']).size().reset_index(name='Count_Extension')

        # Group by PN(BOM Code/Item) and Date Of Manufacture for 'refresh' category
        refresh_count = extrait_refresh[extrait_refresh['Category'] == 'refresh'].groupby(['PN(BOM Code/Item)', 'Date Of Manufacture','NEName']).size().reset_index(name='Count_Refresh')

        # Merge the grouped data back into extrait_refresh
        extrait_refresh = extrait_refresh.merge(new_count, on=['PN(BOM Code/Item)', 'Date Of Manufacture','NEName'], how='left')
        extrait_refresh = extrait_refresh.merge(extension_count, on=['PN(BOM Code/Item)', 'Date Of Manufacture','NEName'], how='left')
        extrait_refresh = extrait_refresh.merge(refresh_count, on=['PN(BOM Code/Item)', 'Date Of Manufacture','NEName'], how='left')

   
        extrait_refresh.fillna(0, inplace=True)
       
         # Classification of statuses
        extrait_refresh['Status'] = np.where(
            (extrait_refresh['Count_New'] > 0) & 
            (extrait_refresh['Count_Refresh'] == 0) & 
            (extrait_refresh['Count_Extension'] == 0), 'Added Only',
            
            np.where(
                (extrait_refresh['Count_Refresh'] > 0) & 
                (extrait_refresh['Count_New'] == 0) & 
                (extrait_refresh['Count_Extension'] == 0), 'Dismantled Only',
                
                np.where(
                    (extrait_refresh['Count_Extension'] > 0) & 
                    (extrait_refresh['Count_Refresh'] == 0) & 
                    (extrait_refresh['Count_New'] == 0), 'Extension Only',
                    
                    np.where(
                        (extrait_refresh['Count_Refresh'] > 0) & 
                        (extrait_refresh['Count_Extension'] > 0) & 
                        (extrait_refresh['Count_New'] > 0), 'Both',
                        
                        'Not Classified'  # For cases where none of the above conditions are met
                    )
                )
            )
        )
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Final  Inventory Tracking  ")
            st.dataframe(extrait_refresh[['Board Name','Date Of Manufacture','PN(BOM Code/Item)','Count_New','Count_Extension','Count_Refresh','Status']])
             # Button to generate the report
            if st.button('Generate Report'):
                generate_report(extrait_refresh, filename="final_report.pdf")
                st.success("Report generated successfully!")
            if st.button('Save to Excel'):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    extrait_refresh.to_excel(writer, sheet_name='Inventory Tracking', index=False)
                output.seek(0)  # Move cursor to the start of the BytesIO object
                st.download_button(
                    label="Download Excel file",
                    data=output.getvalue(),
                    file_name="final_inventory_tracking.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                    
        with col2:
           
        
            totals = {
                    'Nouveaux': extrait_refresh['Count_New'].sum(),
                    'Extension': extrait_refresh['Count_Extension'].sum(),
                    'Rafraîchissement': extrait_refresh['Count_Refresh'].sum()
                }
                
                
            totals_df = pd.DataFrame(list(totals.items()), columns=['Catégorie', 'Total'])
            
            fig_totals_pie = px.pie(
                    totals_df,
                    names='Catégorie',
                    values='Total',
                    title=f'Distribution of category items ',
                    labels={'Catégorie': 'Catégorie', 'Total': 'Nombre de Tableaux'}
                )
                
            st.plotly_chart(fig_totals_pie)

        extrait_refresh['Date Of Manufacture'] = pd.to_datetime(extrait_refresh['Date Of Manufacture'], errors='coerce')
        year_options = extrait_refresh['Date Of Manufacture'].dt.year.unique()
        selected_year = st.sidebar.selectbox('Select Year', sorted(year_options))
          
        extrait_refresh_year = extrait_refresh[extrait_refresh['Date Of Manufacture'].dt.year == selected_year]
        col1, col2= st.columns(2)
        with col1: 
        
            st.write(f"### Boards for the year {selected_year} ")
            st.dataframe(extrait_refresh_year[['Board Name', 'Date Of Manufacture', 'PN(BOM Code/Item)', 'Category', 'Count_New', 'Count_Extension', 'Count_Refresh']])
        with col2:
         # Displaying the message and button in the same line
            st.write("If you want to save the inventory for the selected year, click the 'Save to Excel' button: ", end="")
            # Use a unique key for the button
            if st.button('Save ', key='save_to_excel'):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    extrait_refresh.to_excel(writer, sheet_name='Inventory Tracking', index=False)
                output.seek(0)  # Move cursor to the start of the BytesIO object
                st.download_button(
                    label="Download Excel file",
                    data=output.getvalue(),
                    file_name="final_inventory_tracking_for_selected_year.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

            melted_data = extrait_refresh_year.melt(id_vars=['Board Name', 'PN(BOM Code/Item)'], 
                                                value_vars=['Count_New', 'Count_Extension', 'Count_Refresh'],
                                                var_name='Category',
                                                value_name='Count')

            
            fig_board_histogram = px.bar(
                    melted_data,
                    x='Board Name',
                    y='Count',
                    color='Category',
                    text='PN(BOM Code/Item)',  # Affiche les codes PN comme labels
                    title=f'Histogramme  of items on {selected_year}',
                    labels={'Board Name': 'boardnawe with serial number', 'Count': 'count', 'Category': 'Category'},
                    barmode='stack'  
                )

            
            fig_board_histogram.update_traces(texttemplate='%{text}', textposition='inside')

            st.plotly_chart(fig_board_histogram)
               
        pn_options = extrait_refresh['PN(BOM Code/Item)'].unique()

        selected_pn = st.sidebar.selectbox(
            "Select PN(BOM Code/Item) ",
            pn_options
        )

        if selected_pn:
            filtered_by_pn = extrait_refresh[extrait_refresh['PN(BOM Code/Item)'] == selected_pn]

            # Creating a row with two columns for the details and visualization
            col1, col2 = st.columns([3, 1])  # Adjust sizes as needed

            with col1:
                st.write(f"### Details for PN(BOM Code/Item): {selected_pn}")
                st.dataframe(filtered_by_pn)

            with col2:
                st.write("### Visualization")
                # Check for the 'Date Of Manufacture' column and create the visualization
                if 'Date Of Manufacture' in filtered_by_pn.columns:
                    # Extract year from Date Of Manufacture for better visualization
                    filtered_by_pn['Year'] = pd.to_datetime(filtered_by_pn['Date Of Manufacture'], errors='coerce').dt.year

                    # Create a bar chart to show the count of items by year of manufacture
                    year_counts = filtered_by_pn['Year'].value_counts().reset_index()
                    year_counts.columns = ['Year', 'Count']

                    # Create a bar plot using Plotly
                    fig = px.bar(
                        year_counts,
                        x='Year',
                        y='Count',
                        title=f'Count of Items for PN {selected_pn} by Year of Manufacture',
                        labels={'Count': 'Number of Items', 'Year': 'Year'},
                        color='Count',  # Color by count for better visualization
                        text='Count'  # Display count on bars
                    )

                    # Update layout for better readability
                    fig.update_layout(xaxis_title='Year', yaxis_title='Number of Items')

                    # Show the plot
                    st.plotly_chart(fig)
        else:
            st.write("Please select or enter a PN(BOM Code/Item) to view details.")

    


# Function to load the data from a file
def read_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        excel_df = pd.read_excel(file)
        csv_data = StringIO()
        excel_df.to_csv(csv_data, index=False)
        csv_data.seek(0)
        return pd.read_csv(csv_data)

# Visualization function for board data
def visualize_data(filtered_by_date, df):
    # Bar Chart: Count of Board Types for filtered data
    if not filtered_by_date.empty:
   
        # Pie Chart: Distribution of Board Types in the overall dataset
        board_type_counts_overall = df['Board Name'].value_counts().reset_index()
        board_type_counts_overall.columns = ['Board Name', 'Count']
        fig_pie = px.pie(board_type_counts_overall, values='Count', names='Board Name', title='Board Names Distribution')
        st.plotly_chart(fig_pie)

    else:
        st.write("No data available for the selected filters.")
def visualize_datapm(filtered_by_date):
    # Check if the filtered data is not empty
    if not filtered_by_date.empty:
        # Create tuples of (Board Name, PN(BOM Code/Item)) for visualization
        filtered_by_date['Board_PN_Tuple'] = filtered_by_date.apply(
            lambda row: f"{row['Board Name']} ({row['PN(BOM Code/Item)']})", axis=1
        )

        # Count occurrences of each tuple
        tuple_counts = filtered_by_date['Board_PN_Tuple'].value_counts().reset_index()
        tuple_counts.columns = ['Board_PN_Tuple', 'Count']

        # Create a bar chart for the counts of Board Name and PN BOM Code pairs
        fig_combined = px.bar(
            tuple_counts,
            x='Board_PN_Tuple',
            y='Count',
            title='Count of Boards by (Board Name, PN(BOM Code/Item))',
            labels={'Board_PN_Tuple': 'Board Name (PN(BOM Code/Item))', 'Count': 'Count'},
            color='Count',
            text='Count'  # Display the counts clearly on the bars
        )

        # Update the layout for better readability
        fig_combined.update_layout(
            xaxis_title='(Board Name, PN(BOM Code/Item))',
            yaxis_title='Number of Boards',
            xaxis_tickangle=-45,
            xaxis_tickmode='linear',  # Ensures proper spacing for each label
            uniformtext_minsize=8,  # Ensures minimum font size for text
            uniformtext_mode='hide'  # Hides text if too small to display properly
        )

        # Customize the appearance of the text on the bars to be clearer
        fig_combined.update_traces(
            texttemplate='%{text}',  # Only show the number (count) as text
            textposition='outside',  # Place the text above the bars for better visibility
            marker=dict(color='blue')  # Optionally, you can set a custom color for the bars
        )

        # Show the plot
        st.plotly_chart(fig_combined)

    else:
        st.write("No data available for the selected filters.")



boards_data = {
            "FModule": "",
            "GTMU": "Transmission processing 2G GSM (No Bandwidth)",
            "FAN": "Cards are found at all the sites (No Bandwidth)",
            "UPEU": "Power module",
            "UMPT": "Unité Principale de Traitement et de Transmission Universelle",
            "WMPT": "Transmission processing 3G WCDMA",
            "LMPT": "Transmission processing 4G LTE",
            "UBBP": "Process uplink and downlink baseband signals",
            "WBBP": "Processes uplink and downlink baseband signals 3G",
            "LBBP": "Processes uplink and downlink baseband signals 4G",
            "MRRU": "The frequency bands can vary depending on network specifications",
            "RFU": "The frequency bands can vary depending on network specifications",
            "AUU": "The frequency bands can vary depending on network specifications",
            "PSU": "Used for power supply (No Bandwidth)",
            "PDB": "Used for power supply (No Bandwidth)",
            "PMI": "Used for power supply (No Bandwidth)",
            "TCU": "Used for control and synchronization (No Bandwidth)"
        }

# Main dashboard function
def dashboard():
    st.title("Global Inventory Dashboard")

    # File upload by the user
    file = st.file_uploader("Upload your inventory file", type=["csv", "xlsx"])
    if file is not None:
        df = read_file(file)

        # Check if the DataFrame is empty
        if df.empty:
            st.write("The file is empty or could not be processed.")
            return

        # Extract necessary columns and add 'Band Unit'
        required_columns = ['NEType', 'NEFdn', 'NEName', 'Board Name', 'Board Type', 'PN(BOM Code/Item)', 'Manufacturer Data', 'Date Of Manufacture']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing columns in the file: {', '.join(missing_columns)}")
            return
        else:
            # Select necessary columns and add 'Band Unit'
            df['Date Of Manufacture'] = pd.to_datetime(df["Date Of Manufacture"], errors='coerce')
            data_extracted = df[required_columns]
            data_extracted['Band Unit'] = df['Manufacturer Data'].apply(lambda x: extract_band_info(x)[0])

        # Global overview: Summary statistics
        st.header("Global Overview")
        total_boards = data_extracted['Board Name'].nunique()
        total_board_types = data_extracted['Board Type'].nunique()
        total_boards_code = data_extracted['PN(BOM Code/Item)'].nunique()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Boards Names", total_boards)
        col2.metric("Total Board Types", total_board_types)
        col3.metric("Total  boards PN(BOM Code/Item) ",  total_boards_code)
             # Filter by NEName (Network Element Name)
        
        st.sidebar.header("Select NEName (Zone)")
        unique_nenames = data_extracted['NEName'].unique()
        selected_nename = st.sidebar.selectbox("Select a NEName", unique_nenames)

        # Filter the data for the selected NEName
        filtered_data = data_extracted[data_extracted['NEName'] == selected_nename]

        if filtered_data.empty:
            st.write(f"No boards found for {selected_nename}")
        else:
             
            # Combine the board names with their features from the dictionary
            filtered_data['Board Feature'] = filtered_data['Board Name'].apply(lambda x: boards_data.get(x, "Feature not found"))

            # Histogram: Number of Boards in the Selected Zone with Features
            st.subheader(f"Number of Boards in {selected_nename} by Name (with Features)")

            # Prepare data for the histogram
            board_count_by_name = filtered_data['Board Name'].value_counts().reset_index()
            board_count_by_name.columns = ['Board Name', 'Count']

            # Merge the features into the count data for display in the chart
            board_count_by_name['Feature'] = board_count_by_name['Board Name'].apply(lambda x: boards_data.get(x, "Feature not found"))

            # Create the bar chart with hover text for features
            fig = px.bar(
                board_count_by_name,
                x='Board Name',
                y='Count',
                title=f'Number of Boards in {selected_nename} (with Features)',
                labels={'Count': 'Number of Boards'},
                hover_data={'Feature': True}  # Add features as hover data
            )

            # Show the plot
            st.plotly_chart(fig)

        # Add filters for date period
        st.sidebar.header("Filter by Date")
        start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
        end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))
        if start_date > end_date:
            st.error("Start date cannot be after end date.")
            return

        # Filter the data based on the selected date range
        filtered_by_date = data_extracted[(data_extracted['Date Of Manufacture'] >= pd.to_datetime(start_date)) & 
                                          (data_extracted['Date Of Manufacture'] <= pd.to_datetime(end_date))]

        # Visualization: Distribution of board types
        st.subheader(f"Board Name Distribution from {start_date} to {end_date}")
        visualize_datapm(filtered_by_date)
                # Visualization: Band Unit distribution
        st.subheader(f"Band Unit Distribution from {start_date} to {end_date}")

        # Ensure 'Band Unit' is extracted correctly from 'Manufacturer Data'
        filtered_by_date['Band Unit'] = filtered_by_date['Manufacturer Data'].apply(lambda x: extract_band_info(x)[0])

        # Create a DataFrame for the Band Unit distribution
        band_unit_distribution = filtered_by_date.groupby(['Band Unit', 'Board Name']).size().reset_index(name='Count')

        # Create the Band Unit distribution pie chart
        fig_band_unit = px.pie(
            band_unit_distribution,
            names='Band Unit',
            values='Count',
            title='Distribution of Band Units with Board Names',
            hover_data=['Board Name'],  # Show Board Names on hover
        )

                
        # Create columns for displaying the chart and DataFrame side by side
        col1, col2 = st.columns(2)

        # Display the pie chart in the first column
        with col1:
            st.plotly_chart(fig_band_unit)

        # Display the DataFrame in the second column
        with col2:
            st.subheader("Filtered Data with Band Unit:")
            st.write(filtered_by_date[['NEName', 'Board Name', 'PN(BOM Code/Item)', 'Date Of Manufacture', 'Band Unit']])

        # Display extracted data
        #st.subheader("Filtered Data:")
        #st.write(filtered_by_date)

        # Show detailed board information
        #st.subheader("Detailed Board Information")
        #st.write(filtered_by_date[['NEName', 'Board Name', 'Board Type', 'PN(BOM Code/Item)', 'Date Of Manufacture', 'Band Unit']])
         

        