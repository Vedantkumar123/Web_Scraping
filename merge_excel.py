import pandas as pd
import glob

# Specify the folder path
folder_path = 'Output_Files'

# Use glob to get all Excel files in the folder
excel_files = glob.glob(f'{folder_path}/*.xlsx')

# Initialize an empty list to hold the dataframes
dataframes = []

# Loop through the list of Excel files and read each one into a dataframe
for file in excel_files[0:2]:
    df = pd.read_excel(file)
    dataframes.append(df)
    print(file)

# Concatenate all dataframes in the list
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined dataframe to a new Excel file
combined_df.to_excel('C:/Users/KIIT/OneDrive/Desktop/Vedant_Official/Vindhaya_work/web_scrap/combined_file.xlsx', index=False)