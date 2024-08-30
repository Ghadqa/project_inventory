from io import StringIO
import pandas as pd
import plotly.express as px
import streamlit as st 
import numpy as np
import re
from io import BytesIO  



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

def read_file(file):
            if file.name.endswith('.csv'):
                return pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                excel_df = pd.read_excel(file)
                csv_data = StringIO()
                excel_df.to_csv(csv_data, index=False)
                csv_data.seek(0)  #
                return pd.read_csv(csv_data)
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

   
        newsite = df2[~df2['NEName'].isin(df1['NEName'])]
        st.write("### New Site")
        st.write(newsite)

   
        rru_info = newsite[newsite['Board Name'].str.contains('RRU', na=False)]
        non_rru_info = newsite[~newsite['Board Name'].str.contains('RRU', na=False)]

    
        unique_nenames = newsite['NEName'].unique()
        selected_name = st.sidebar.selectbox("Select NEName to display details for new site ", unique_nenames)

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

        new_boards = df2[~df2['PN(BOM Code/Item)'].isin(df1['PN(BOM Code/Item)'])]

    
        old_sites = df1[~df1['NEName'].isin(df2['NEName'])]

        old_boards = df1[~df1['PN(BOM Code/Item)'].isin(df2['PN(BOM Code/Item)'])]

        old_boards['Band_Info'] = old_boards['Manufacturer Data'].apply(lambda x: extract_band_info(x)[0] if x else None)
        extrait_old = old_boards[['Board Name', 'PN(BOM Code/Item)', 'Band_Info', 'Date Of Manufacture']]
        st.write("### Old Items Information")
        st.write(extrait_old)

        new_boards_newsite = new_boards[new_boards['NEName'].isin(newsite['NEName'])]

   
        dismantled_boards = df2[df2['PN(BOM Code/Item)'].isin(df1['PN(BOM Code/Item)'])]
        dismantled_boards_info = dismantled_boards[['Board Name', 'PN(BOM Code/Item)', 'Date Of Manufacture', 'Manufacturer Data']].drop_duplicates()
        dismantled_boards_info['Band_Info'] = dismantled_boards_info['Manufacturer Data'].apply(lambda x: extract_band_info(x)[0] if x else None)
        st.write("### dismantled  Items Information")
        st.write(dismantled_boards_info)
        newsite['Band_Info'] = newsite['Manufacturer Data'].apply(lambda x: extract_band_info(x)[0] if x else None)

        newsite['Category'] = np.nan

        newsite.loc[newsite['NEName'].isin(new_boards_newsite['NEName']), 'Category'] = 'new'

        #
        rru_boards = newsite[newsite['Board Name'].str.startswith(('RRU', 'MRRU', 'LRRU'))]

        def update_category_for_board(row, matching_rrus, category):
            """
            Updates the category of a board based on frequency bands.

            Args:
            - row: Current RRU board row.
            - matching_rrus: DataFrame containing matching old RRU boards.
            - category: Category to assign ('refresh' or 'extension').
            """
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

        st.write("### Final New Site Data with Categories")
        extrait_refresh=newsite[['NEName', 'PN(BOM Code/Item)','Board Name', 'Date Of Manufacture', 'Category']]
        extrait_refresh['Description'] = extrait_refresh['Board Name'].map(boards_data)

     

        
        st.write(extrait_refresh)
     
        selected_site = st.sidebar.selectbox(
            "Select Site Name (NEName) to view details for Final New Sites Data with Categories of Boards",
            extrait_refresh['NEName'].unique()
        )

        if selected_site:
            
            site_data = extrait_refresh[extrait_refresh['NEName'] == selected_site]
            
            st.write(f"### Boards for Site: {selected_site}")
          
               
            new_count =site_data[site_data['Category'] == 'new'].groupby('PN(BOM Code/Item)').size().reset_index(name='Count_New')
            extension_count = site_data[site_data['Category'] == 'extension'].groupby('PN(BOM Code/Item)').size().reset_index(name='Count_Extension')
            refresh_count =site_data[site_data['Category'] == 'refresh'].groupby('PN(BOM Code/Item)').size().reset_index(name='Count_Refresh')
            site_data = site_data.merge(new_count, on='PN(BOM Code/Item)', how='left')
            site_data = site_data.merge(extension_count, on='PN(BOM Code/Item)', how='left')
            site_data = site_data.merge(refresh_count, on='PN(BOM Code/Item)', how='left')
            
            
            
            site_data['Status'] = np.where(
                (site_data['Count_New'] > 0) & (site_data['Count_Refresh'] == 0) & (site_data['Count_Extension'] == 0), 'Added Only',
                np.where(
                    (site_data['Count_Refresh'] >0) & (site_data['Count_Extension'] > 0), 'Dismantled Only',
                    'Both'
                )
            )
            site_data['Status_Numeric'] = site_data['Status'].map({
                'Added Only': 1,
                'Dismantled Only': 1,
                'Both': 2
            })
            status_counts = site_data['Status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
        
            st.dataframe(site_data[['Board Name','Date Of Manufacture','PN(BOM Code/Item)', 'Category','Count_New','Count_Extension','Count_Refresh']])
          
                
            totals = {
                'Nouveaux': site_data['Count_New'].sum(),
                'Extension': site_data['Count_Extension'].sum(),
                'Rafraîchissement': site_data['Count_Refresh'].sum()
            }
            
            
            totals_df = pd.DataFrame(list(totals.items()), columns=['Catégorie', 'Total'])
          
            fig_totals_pie = px.pie(
                totals_df,
                names='Catégorie',
                values='Total',
                title=f'Distribution of category items in {selected_site}',
                labels={'Catégorie': 'Catégorie', 'Total': 'Nombre de Tableaux'}
            )
            
            st.plotly_chart(fig_totals_pie)
            site_data['Date Of Manufacture'] = pd.to_datetime(site_data['Date Of Manufacture'], errors='coerce')
            year_options = site_data['Date Of Manufacture'].dt.year.unique()
            selected_year = st.sidebar.selectbox('Select Year', sorted(year_options))
          
            site_data_year = site_data[site_data['Date Of Manufacture'].dt.year == selected_year]
            st.write(f"### Boards for the year {selected_year} in {selected_site} ")
            st.dataframe(site_data_year[['Board Name', 'Date Of Manufacture', 'PN(BOM Code/Item)', 'Category', 'Count_New', 'Count_Extension', 'Count_Refresh']])
          
            melted_data = site_data_year.melt(id_vars=['Board Name', 'PN(BOM Code/Item)'], 
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
               
        pn_options = site_data['PN(BOM Code/Item)'].unique()

     
        selected_option = st.sidebar.selectbox(
            "Select PN(BOM Code/Item)on Final New Sites Data with Categories of Boards",
            ["Select from list"] + list(pn_options)  
        )

        if selected_option == "Select from list":
          
            manual_pn = st.sidebar.text_input("Enter PN(BOM Code/Item):", "")
            selected_pn = manual_pn if manual_pn else None
        else:
            selected_pn = selected_option

       
        if selected_pn:
            filtered_by_pn = site_data[site_data['PN(BOM Code/Item)'] == selected_pn]

            st.write(f"### Details for PN(BOM Code/Item): {selected_pn}")
            st.dataframe(filtered_by_pn)
        else:
            st.write("Please select or enter a PN(BOM Code/Item) to view details.")
        
                

                    

                

                    
                            

                                    

