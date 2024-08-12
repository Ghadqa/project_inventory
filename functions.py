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
from datetime  import datetime

def show_compare():
    st.subheader("Upload files for comparison")

    # Upload first file
    file1 = st.file_uploader("Upload old data inventory (CSV or Excel file)", type=["csv", "xlsx"], key="file1")
    
    # Upload second file
    file2 = st.file_uploader("Upload new data inventory (CSV or Excel file)", type=["csv", "xlsx"], key="file2")

    if file1 is not None and file2 is not None:
        # Read the files into DataFrames
        df1 = pd.read_csv(file1) if file1.name.endswith('.csv') else pd.read_excel(file1)
        df2 = pd.read_csv(file2) if file2.name.endswith('.csv') else pd.read_excel(file2)

        # Vérification des colonnes requises
        required_columns = ["Board_Name", "Date_Of_Manufacture", "date_of_integration", "sn_bar_code", "NEName"]
        missing_columns = [col for col in required_columns if col not in df1.columns or col not in df2.columns]

        if missing_columns:
            st.error(f"The following columns are missing in the file(s): {', '.join(missing_columns)}")
            return
        
        # Comparer les colonnes "NEName"
        common_df, diff_file1_df, diff_file2_df = compare_nename_columns(df1, df2, "NEName")
        common_values = set(common_df["NEName"])
       
        # Affichage des résultats avec mise en évidence
        if common_df.empty:
            st.write("### New sites in file2")
            st.write(set(diff_file2_df["NEName"]))
        else:
            st.write("### Same sites in both files")
            st.dataframe(common_df.style.applymap(lambda x: 'background-color: lightgreen'))

        # Affichage des lignes avec mise en évidence
        st.write("### All information in First File")
        st.dataframe(highlight_common_rows(df1, common_values, "NEName"))

        st.write("### All information in Second File")
        st.dataframe(highlight_common_rows(df2, common_values, "NEName"))

        # Convert date columns to datetime
        df1['Date_Of_Manufacture'] = pd.to_datetime(df1['Date_Of_Manufacture'], errors='coerce')
        df2['Date_Of_Manufacture'] = pd.to_datetime(df2['Date_Of_Manufacture'], errors='coerce')

        # Select filter option
        filter_option = st.sidebar.selectbox("Choose date filter type", ["Single Date", "Date Range"])

        if filter_option == "Single Date":
            # Date selection
            selected_date = st.sidebar.date_input("Select a date", min_value=datetime(2000, 1, 1), max_value=datetime(2100, 12, 31))
            filtered_df1 = df1[df1['Date_Of_Manufacture'] < pd.to_datetime(selected_date)]
            filtered_df2 = df2[df2['Date_Of_Manufacture'] < pd.to_datetime(selected_date)]
            
        elif filter_option == "Date Range":
            # Date range selection
            start_date = st.sidebar.date_input("Select start date", min_value=datetime(2000, 1, 1), max_value=datetime(2100, 12, 31))
            end_date = st.sidebar.date_input("Select end date", min_value=datetime(2000, 1, 1), max_value=datetime(2100, 12, 31))
            filtered_df1 = df1[(df1['Date_Of_Manufacture'] >= pd.to_datetime(start_date)) & (df1['Date_Of_Manufacture'] <= pd.to_datetime(end_date))]
            filtered_df2 = df2[(df2['Date_Of_Manufacture'] >= pd.to_datetime(start_date)) & (df2['Date_Of_Manufacture'] <= pd.to_datetime(end_date))]
        
        if common_df.empty: 
            for value in diff_file2_df['NEName']:
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"### Information boards in new site for NEName = {value} in File 2")
                        filtered2 = df2[df2["NEName"] == value][["NEName", "Board_Name", "Date_Of_Manufacture", "sn_bar_code"]]
                        
                        # Utilisation de la fonction pour mettre en évidence et extraire les lignes surlignées
                        highlighted_df2, styled_df2 = highlight_date_filtered_rows(filtered2, filtered_df2)
                        
                        st.dataframe(styled_df2)  # Afficher le DataFrame stylisé
                       # Afficher uniquement les lignes surlignées
                        
                    with col2:    
                        visualize_data(highlighted_df2,filtered2["NEName"])
        else:            
            for value in common_df['NEName']:
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"### Information boards for NEName = {value} in File 1")
                        filtered = df1[df1["NEName"] == value][["NEName", "Board_Name", "Date_Of_Manufacture", "sn_bar_code"]]
                        
                        # Utilisation de la fonction pour mettre en évidence et extraire les lignes surlignées
                        highlighted_df, styled_df = highlight_date_filtered_rows(filtered, filtered_df1)
                    
                        st.dataframe(styled_df)  # Afficher le DataFrame stylisé
                        #st.write(highlighted_df)  # Afficher uniquement les lignes surlignées
                        visualize_data(highlighted_df,  filtered["NEName"])
                    
                    with col2:    
                        st.write(f"### Information boards for NEName = {value} in File 2")
                        filtered2 = df2[df2["NEName"] == value][["NEName", "Board_Name", "Date_Of_Manufacture", "sn_bar_code"]]
                        
                        # Utilisation de la fonction pour mettre en évidence et extraire les lignes surlignées
                        highlighted_df2, styled_df2 = highlight_date_filtered_rows(filtered2, filtered_df2)
                        
                        st.dataframe(styled_df2)  # Afficher le DataFrame stylisé
                        #st.write(highlighted_df2)  # Afficher uniquement les lignes surlignées
                        visualize_data(highlighted_df2,  filtered2["NEName"])

        st.write("### New boards with more recent manufacture date in File 2")
        post_dates = compare_dates_later(df1, df2, "Date_Of_Manufacture")
        st.write(post_dates)


def show_dashboard1():
    uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True)

    if uploaded_files:
        # Combine all uploaded CSV files into a single DataFrame
        dfs = [pd.read_csv(file) for file in uploaded_files]
        df = pd.concat(dfs, ignore_index=True)

        st.write("Data from uploaded files:")
        st.write(df.head())  # Display the first few rows of the combined DataFrame

        if df.empty:
            st.warning("The uploaded files are empty or couldn't be processed.")
            return

        # Filter data based on board name
        board = st.sidebar.selectbox("Select the board:", options=df["Board_Name"].unique())
        filtered_by_board = df[df["Board_Name"] == board]
        st.write(f"Networks information for selected board {board}")
        st.write(filtered_by_board)

        # Filter data by date of manufacture
        dates = st.sidebar.date_input("Select the date of manufacture:")
        selected_year = dates.year

        # Ensure Date_Of_Manufacture is in datetime format
        df["Date_Of_Manufacture"] = pd.to_datetime(df["Date_Of_Manufacture"], errors='coerce')
        filtered_by_date = df[df["Date_Of_Manufacture"].dt.year == selected_year]
        st.write(f"Boards information for date of manufacture {selected_year}")
        st.write(filtered_by_date)
        
        # Visualize data (assuming visualize_data function is defined)
        visualize_data(filtered_by_date, df)

        # Filter by NEName
        nename = st.sidebar.selectbox("Select the NEName:", options=df["NEName"].unique())
        filtered_by_nename = df[df["NEName"] == nename]
        st.write(f"Boards information for selected network name {nename}")
        st.write(filtered_by_nename)

        if not filtered_by_nename.empty:
            board_name = st.sidebar.selectbox(f"Select the board name for {nename}:", options=filtered_by_nename["Board_Name"].unique())
            
            df_filtered = filtered_by_nename[filtered_by_nename["Board_Name"] == board_name]
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
            
            if board_name in boards_data:
                st.info(f"Information for Board Name {board_name}: {boards_data[board_name]}")
            
            if not df_filtered.empty:
                st.write(f"Manufacturing date for {board_name}:")
                st.write(df_filtered[['Board_Name', 'sn_bar_code', 'Date_Of_Manufacture']])
            else:
                st.write("No data available for the selected board.")
    else:
        st.info("Please upload CSV files to proceed.")


import os 
def send_email(subject, body, to_email, attachment_path=None):
    from_email = "ghadamhadhbi3@gmail.com"
    password = "gh58242543ADA"

    # Create a MIME message
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach the email body to the message
    msg.attach(MIMEText(body, 'plain'))

    # Attach a file (if provided)
    if attachment_path:
        if os.path.isfile(attachment_path):
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename={os.path.basename(attachment_path)}'
                )
                msg.attach(part)
        else:
            print(f"Attachment file not found: {attachment_path}")
            return

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(from_email, password)
            server.send_message(msg)
            print("Email sent successfully!")
    except smtplib.SMTPException as e:
        print(f"Failed to send email. Error: {e}")


def generate_report(data, filename="rapport.pdf"):
    # Créer un modèle HTML pour le rapport
    html_template ="""<!DOCTYPE html>
<html>
<head>
    <title>Report</title>
    <style>
        table { width: 100%; border-collapse: collapse; }
        table, th, td { border: 1px solid black; }
        th, td { padding: 8px; text-align: left; }
    </style>
</head>
<body>
    <h1>Comparison Report</h1>
    <h2>Summary</h2>
    <p>The report presents the results of the comparison of manufacturing dates.</p>
    <h2>Data</h2>
    <table>
        <thead>
            <tr>
                <th>Site name </th>
                <th>Board Name</th>
                <th>Date Of Manufacture</th>
                <th>SN Bar Code</th>
             
            </tr>
        </thead>
        <tbody>
            {% for row in data %}
            <tr>
                <td>{{ row['NEName'] }}</td>
                <td>{{ row['Board_Name'] }}</td>
                <td>{{ row['Date_Of_Manufacture'] }}</td>
                <td>{{ row['sn_bar_code'] }}</td>
               
            </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>
"""
    template = Template(html_template)
    html_report = template.render(data=data.to_dict(orient='records'))

    # Convertir HTML en PDF
    pdfkit.from_string(html_report, filename)

def visualize_comparison(existing_entries, new_entries):
    if existing_entries.empty and new_entries.empty:
        st.warning("No data available for comparison.")
        return

    # Define plot styles
    scatter_style = dict(
        mode='markers',
        marker=dict(size=10, opacity=0.8)
    )

    

    # Combined scatter plot
    if not existing_entries.empty and not new_entries.empty:
        combined_df = pd.concat([existing_entries.assign(type='Existing'), new_entries.assign(type='New')])
        fig_combined = px.scatter(
            combined_df,
            x='Date_Of_Manufacture',
            y='Date_Of_Manufacture',
            color='type',
            symbol='type',
            title="Combined Comparison of Manufacturing Dates",
            labels={
                'Date_Of_Manufacture': "Date of Manufacture",
                'type': "Entry Type"
            },
            color_discrete_map={'Existing': 'blue', 'New': 'red'},
            symbol_map={'Existing': 'circle', 'New': 'x'}
        )
        fig_combined.update_traces(**scatter_style)
        fig_combined.update_layout(
            xaxis_title="Date of Manufacture",
            yaxis_title="Date of Manufacture",
            legend_title="Entry Type"
        )
        st.plotly_chart(fig_combined, use_container_width=True)


# Initialize the session state to store new entries
if 'new_networks' not in st.session_state:
    st.session_state['new_networks'] = []
if 'new_boards' not in st.session_state:
    st.session_state['new_boards'] = []

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    st.success(f"Les données ont été enregistrées dans {filename}.")

def read_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format.")
        return pd.DataFrame()

def get_data_from_uploaded_files(uploaded_files):
    if uploaded_files:
        try:
            db_file = "inventorydata.db"
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            st.success('Opened database successfully')

            df_list = [read_file(uploaded_file) for uploaded_file in uploaded_files]
            combined_df = pd.concat(df_list, ignore_index=True)

            if not combined_df.empty:
                for _, row in combined_df.iterrows():
                    cursor.execute('''
                        SELECT id FROM networks WHERE NEName = ? AND NEType = ? AND NEFdn = ?
                    ''', (row['NEName'], row['NEType'], row['NEFdn']))
                    result = cursor.fetchone()

                    if result is None:
                        cursor.execute('''
                            INSERT INTO networks (NEName, NEType, NEFdn)
                            VALUES (?, ?, ?)
                        ''', (row['NEName'], row['NEType'], row['NEFdn']))
                        network_id = cursor.lastrowid
                    else:
                        network_id = result[0]

                    cursor.execute('''
                        INSERT INTO boards (network_id, Board_Name, Board_Type, date_of_integration, Date_Of_Manufacture, sn_bar_code)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (network_id, row['Board Name'], row['Board Type'], row['Date of Integration'], row['Date Of Manufacture'], row['PN(BOM Code/Item)']))

                    board_id = cursor.lastrowid

                    cursor.execute('''
                        INSERT INTO network_boards (network_id, board_id)
                        VALUES (?, ?)
                    ''', (network_id, board_id))

                conn.commit()
                st.success('Data uploaded successfully')
            else:
                st.warning("The uploaded files are empty.")
        finally:
            conn.close()
    else:
        st.warning("No files uploaded.")

    return combined_df

def load_data(conn):
    networks = pd.read_sql_query("SELECT * FROM networks", conn)
    boards = pd.read_sql_query("SELECT * FROM boards", conn)
    network_boards = pd.read_sql_query("SELECT * FROM network_boards", conn)
    return networks, boards, network_boards

def compare_data(old_data, new_data):
    if isinstance(new_data, dict):
        new_board_df = pd.DataFrame([new_data])
    else:
        new_board_df = pd.DataFrame(new_data)

    old_boards_df = pd.DataFrame(old_data)

    if new_board_df.empty or old_boards_df.empty:
        st.error("No data to compare.")
        return

    required_columns = ['sn_bar_code', 'NEName']
    if not all(col in old_boards_df.columns for col in required_columns) or not all(col in new_board_df.columns for col in required_columns):
        st.error("Required columns are missing in the data.")
        return

    merged = pd.merge(new_board_df, old_boards_df, on=required_columns, suffixes=('_new', '_old'), how='outer', indicator=True)
    st.write("### old and new data ")
    st.dataframe(merged)

    return merged

def is_database_empty(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    if not tables:
        conn.close()
        return True

    for table_name in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name[0]};")
        row_count = cursor.fetchone()[0]
        if row_count > 0:
            conn.close()
            return False

    conn.close()
    return True

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


def visualize_data(filtered_entries1,df_networks):
   
    # Create a bar plot using Plotly
    board_counts = filtered_entries1['Board_Name'].value_counts().reset_index()
    board_counts.columns = ['Board_Name', 'Quantity']

    fig_bar = px.bar(
        board_counts,
        x="Board_Name",
        y="Quantity",
        title=f"Board Quantities by Name ",
        labels={"Board_Name": "Board Name", "Quantity": "Quantity"},
        text="Quantity"
    )
    fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
    fig_bar.update_layout(yaxis_title="Quantity")
    st.plotly_chart(fig_bar, use_container_width=True)

    board_counts = filtered_entries1.groupby(['NEName', 'Board_Name']).size().reset_index(name='Quantity')

    fig_bar = px.bar(
        board_counts,
        x="NEName",
        y="Quantity",
        color="Board_Name",
        title=f"Distribution of Boards by Network Name",
        labels={"NEName": "Network Name", "Quantity": "Quantity", "Board_Name": "Board Name"},
        text="Quantity"
    )
    fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
    fig_bar.update_layout(yaxis_title="Quantity", barmode='stack')
    st.plotly_chart(fig_bar, use_container_width=True)
    
def compare_nename_columns(df1, df2, column_name="NEName"):
    # Trouver les valeurs uniques dans chaque DataFrame
    unique_ne1 = set(df1[column_name].dropna())
    unique_ne2 = set(df2[column_name].dropna())
    
    # Trouver les valeurs communes et différentes
    common_values = unique_ne1.intersection(unique_ne2)
    diff_values_file1 = unique_ne1 - unique_ne2
    diff_values_file2 = unique_ne2 - unique_ne1
    
    # Convertir les ensembles en DataFrames pour affichage
    common_df = pd.DataFrame({"NEName": list(common_values)})
    diff_file1_df = pd.DataFrame({"NEName": list(diff_values_file1)})
    diff_file2_df = pd.DataFrame({"NEName": list(diff_values_file2)})

    return common_df, diff_file1_df, diff_file2_df

def highlight_common_rows(df, common_values, column_name="NEName"):
    # Fonction de mise en forme conditionnelle
    def highlight_row(row):
        # Surligner toute la ligne si la valeur de la colonne spécifiée est dans common_values
        return ['background-color: lightgreen' if row[column_name] in common_values else '' for _ in row]

    # Appliquer la mise en forme conditionnelle sur l'ensemble des lignes
    return df.style.apply(highlight_row, axis=1)


    return df.style.apply(highlight_row, axis=1)
def highlight_date_filtered_rows(df, date_filtered, column_name="NEName"):
    # Fonction pour mettre en évidence les lignes
    def highlight_row(row):
        if row.name in date_filtered.index:
            return ['background-color: red' for _ in row]
        return ['' for _ in row]

    # Extraire uniquement les lignes qui sont dans date_filtered
    highlighted_rows = df[df.index.isin(date_filtered.index)]
    
    # Appliquer le style sur tout le DataFrame, mais uniquement les lignes surlignées seront colorées
    styled_df = df.style.apply(highlight_row, axis=1)
    
    return highlighted_rows, styled_df

     

        # Display filtered data for common NEName values
import openai        

import streamlit as st
import pandas as pd
import openai

def compare_nename_columns(df1, df2, column_name):
    # Ensure both dataframes have the specified column
    if column_name not in df1.columns or column_name not in df2.columns:
        st.error(f"Column '{column_name}' not found in one or both files.")
        return None, None, None
    
    # Find common and differing values
    common_values = pd.merge(df1[[column_name]], df2[[column_name]], on=column_name)
    diff_file1 = df1[~df1[column_name].isin(common_values[column_name])]
    diff_file2 = df2[~df2[column_name].isin(common_values[column_name])]
    
    return common_values, diff_file1, diff_file2

def ask_azure_openai(question, context):
    response = openai.Completion.create(
        engine="inventory_gpt",  # Ensure this matches your deployment name in Azure
        prompt=f"{context}\n\nQuestion: {question}",
        max_tokens=150
    )
    return response.choices[0].text.strip()

def chat():
    st.title("Upload Two Files for Comparison and Ask Questions")

    # File upload section for two files
    uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) == 2:
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
                file1_head = df1.head().to_string(index=False)
                file2_head = df2.head().to_string(index=False)
                file1_summary = df1.describe().to_string()
                file2_summary = df2.describe().to_string()
                # Use the comparison results as context for the question
                context = f"Common NEName values:\n{common_df.to_string(index=False)}\n\n" \
                          f"NEName values in File 1 but not in File 2:\n{diff_file1_df.to_string(index=False)}\n\n" \
                          f"NEName values in File 2 but not in File 1:\n{diff_file2_df.to_string(index=False)}\n\n" \
                          f"First few rows of File 1:\n{file1_head}\n\n" \
                          f"First few rows of File 2:\n{file2_head}\n\n" \
                          f"Summary of File 1:\n{file1_summary}\n\n" \
                          f"Summary of File 2:\n{file2_summary}"
                # Limit context size if necessary
                if len(context) > 4000:  # Adjust based on token limit and size of responses
                    context = context[:4000] + "\n[Content truncated due to size.]"

                answer = ask_azure_openai(question, context)
                st.write(f"Answer: {answer}")

    else:
        st.warning("Please upload exactly two CSV files.")

def format_response(df, query):
    # Filter the dataframe based on the query (e.g., 'NAB_0014_LM')
    result_df = df[df['NEFdn'].str.contains(query, na=False, case=False)]
    
    if not result_df.empty:
        return result_df[['Board_Name', 'NEFdn']].to_string(index=False)
    else:
        return "No matching data found."