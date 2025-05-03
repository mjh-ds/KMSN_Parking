
import pdfplumber, re, os
import pandas as pd
from datetime import datetime


def parse_pdf(pdfBlob):
    with pdfplumber.open(f'/workspaces/data/msn_parking/output/raw/{cFile}') as pdf:
        # Initialize an empty string for the text
        text = ''
        
        # Iterate through all pages and extract text
        for page in pdf.pages:
            text += page.extract_text()

    datetime_str = cFile.split('_')[1] + ' ' + cFile.split('_')[2].replace('.pdf', '')
    dt = datetime.strptime(datetime_str, '%Y-%m-%d %H-%M-%S')

    return(text, dt)

def df_from_text(c_date,textBlob):
    # Extracting data using regex 
    pattern = re.compile(r'(Level \d{1,2} \d{1,3} \d{1,3}|Surface \d{1,3} \d{1,3}|Economy \d{1,3} \d{1,3})\n')
    matches = pattern.findall(textBlob) 

    # Create a DataFrame
    data = []
    for match in matches:
        parts = match.split()
        name = parts[0] + ' ' + parts[1] if parts[0] == 'Level' else parts[0]
        available = int(parts[2] if parts[0] == 'Level' else parts[1])
        total = int(parts[3] if parts[0] == 'Level' else parts[2])
        if name == 'Level 12': name = 'Level 1'
        data.append([name, available, total])

    df = pd.DataFrame(data, columns=['name', 'available_spots', 'total_spots'])
    df.insert(0, 'datetime', c_date)
    return(df)

iteration_count = 0
dfs = []
rawFiles = os.listdir('/workspaces/data/msn_parking/output/raw/')
for cFile in rawFiles:
    print(cFile)
    if iteration_count >= 200000000: 
        break
    # Open the PDF file
    x,y = parse_pdf('/workspaces/data/msn_parking/output/raw/{cFile}')
    temp_df = df_from_text(y,x)
    # Define the path to the CSV file
    csv_file_path = '/workspaces/data/msn_parking/output/commute_clean.csv'
    # Check if the file exists to decide if header is needed
    try:
        with open(csv_file_path, 'x') as f:
            # If file does not exist, write DataFrame with header
            temp_df.to_csv(f, index=False)
    except FileExistsError:
        # If file exists, append without header
        temp_df.to_csv(csv_file_path, mode='a', header=False, index=False)

    iteration_count += 1

    # Source file path
    source = f'/workspaces/data/msn_parking/output/raw/{cFile}'
    destination = f'/workspaces/data/msn_parking/output/cleaned/{cFile}'
    # Copy the file using os.system
    os.system(f'mv {source} {destination}')
    print(f"File copied from {source} to {destination}")



