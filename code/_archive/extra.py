def parse_pdf(cFile):
    with pdfplumber.open(cFile) as pdf:
        # Initialize an empty string for the text
        text = ''
        
        # Iterate through all pages and extract text
        for page in pdf.pages:
            text += page.extract_text()

    fn = os.path.basename(cFile)
    datetime_str = fn.split('_')[1] + ' ' + fn.split('_')[2].replace('.pdf', '')
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







from pathlib import Path

Path("data//test.tar.xz") 



import tarfile
import os

def extract_pdfs_from_tar_xz(tar_xz_file, output_dir):
    """Extracts PDF files one by one from a .tar.xz archive, keeping only the filename."""
    with tarfile.open(tar_xz_file, "r:xz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".pdf"):  # Filter PDFs
                filename = os.path.basename(member.name)  # Extract only the filename
                print(f"Extracting: {filename}")
                
                with tar.extractfile(member) as extracted_file:
                    output_path = f"{output_dir}/{filename}"  # Save using only filename
                    with open(output_path, "wb") as out:
                        out.write(extracted_file.read())
                    print(f"Saved to: {output_path}")

# Example usage
tar_xz_file = "C:/Users/Michael/Documents/GitHub/KMSN_Parking/data/test.tar.xz"
output_dir = "C:/Users/Michael/Documents/GitHub/KMSN_Parking/data/out"

extract_pdfs_from_tar_xz(tar_xz_file, output_dir)




with tarfile.open("C:\\Users\\Michael\\Documents\\GitHub\\KMSN_Parking\\data\\raw.tar.xz", "r:xz") as tar:
        first_member = tar.getmembers()[0]
        with tar.extractfile(first_member) as extracted_file:
            with open("C:\\Users\\Michael\\Documents\\GitHub\\KMSN_Parking\\data\\test.pdf", "wb") as out:
                out.write(extracted_file.read())



def extract_first_from_xz(tar_xz_file, output_file):
    """Extract only the first file from a .tar.xz archive."""
    with tarfile.open(tar_xz_file, "r:xz") as tar:
        first_member = tar.getmembers()[0]  # Get the first file in the archive
        with tar.extractfile(first_member) as extracted_file:
            with open(output_file, "wb") as out:
                out.write(extracted_file.read())

# Example usage
tar_xz_file = "data/raw.tar.xz"  # Replace with your .tar.xz file
output_file = "first_extracted_file.pdf"  # Replace with your desired output file

extract_first_from_xz(tar_xz_file, output_file)
print(f"Extracted {first_member.name} to {output_file}")


















pattern = re.compile(r'(\d{1,3} \d{1,3})\n')
matches = pattern.findall(cr) 
import polars as pl
# Convert string into a list of rows
# Extract only numeric values using regex
numeric_lines = [line for line in cr.split("\n") if re.match(r'^\d+\s+\d+$', line)]  # Keep lines with numbers only

# Convert to a list of rows
rows = [list(map(int, line.split())) for line in numeric_lines]

# Create Polars DataFrame
df = pl.DataFrame(rows, schema=["Column1", "Column2"])



    # Check if the file exists to decide if header is needed
    try:
        with open(csv_file_path, 'x') as f:
            # If file does not exist, write DataFrame with header
            temp_df.to_csv(f, index=False)
    except FileExistsError:
        # If file exists, append without header
        temp_df.to_csv(csv_file_path, mode='a', header=False, index=False)



for cFile in rawFiles:
    print(cFile)

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



for n in range(0,n_total):
    clean_dir(tmp_dir,".pdf")
    print(n)
    c_pdf = extract_nth_file(tar_file, tmp_dir,n)
    cr, dt = crop_and_extract(c_pdf,(442, 179, 574, 561))
    c_df = extract_to_df(cr, dt)

    c_df.write_parquet(f'{tmp_dir}/df_{n}.parquet')


c_pdf = extract_nth_file(tar_file, tmp_dir,6)


cr, dt = crop_and_extract(c_pdf,(442, 179, 574, 561))
extract_to_df(cr, dt)
