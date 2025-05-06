
## Imports ##
#############
import tarfile
import pdfplumber, re, os,shutil,logging
import polars as pl
from datetime import datetime

logging.getLogger("pdfminer").setLevel(logging.ERROR)
##  Functions to use later in the code ##
################################################

# Function to count the number of files in a ZIP folder
def count_files(tar_xz_file):
    """Count the number of files inside a .tar.xz archive."""
    with tarfile.open(tar_xz_file, "r:xz") as tar:
        file_count = sum(1 for member in tar.getmembers() if member.isfile())  # Count only files (not folders)
        print(f"Total number of files: {file_count}")
    return file_count

# Function to extract the nth file from the ZIP folder
def extract_nth_file(tar_xz_file, output_dir, n):
    with tarfile.open(tar_xz_file, "r:xz") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".pdf")] 
        member = members[n] 
        with tar.extractfile(member) as extracted_file:
            filename = os.path.basename(member.name) 
            output_path = os.path.join(output_dir, filename)
            with open(output_path, "wb") as out:
                out.write(extracted_file.read())
    return output_path

# Function to extract the numeric information from the PDF file
def crop_and_extract(document, region):
    # Open the PDF and extract text from the defined region
    with pdfplumber.open(document) as pdf:
        page = pdf.pages[0]  # Choose the first page (modify as needed)
        text = page.crop(region).extract_text()  # Crop to region and extract text
    
    fn = os.path.basename(document)
    datetime_str = fn.split('_')[1] + ' ' + fn.split('_')[2].replace('.pdf', '')
    dt = datetime.strptime(datetime_str, '%Y-%m-%d %H-%M-%S')
    
    return text, dt

# Function to convert the extracted numeric information from the PDF into a dataframe
def extract_to_df(extract, c_date):
    numeric_lines = [line for line in cr.split("\n") if re.match(r'^\d+\s+\d+$', line)]  # Keep lines with numbers only

    # Convert to a list of rows
    rows = [list(map(int, line.split())) for line in numeric_lines]

    # Create Polars DataFrame
    df = pl.DataFrame(rows, schema=["available", "total"], orient="row")
    df = df.with_columns(pl.lit(c_date).alias("datetime"))
    df = df.select(["datetime"] + df.columns[:-1])
    return df

# Function to clean out a directory
def clean_dir(dir, suffix):
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)

        if suffix == "WIPE":
            # Loop through and delete all files
            if os.path.isfile(file_path):  # Ensure it's a file (not a subfolder)
                os.remove(file_path)
                print(f"Deleted: {file_name}")
        else:
            if os.path.isfile(file_path) and file_name.lower().endswith(suffix):
                os.remove(file_path)
                print(f"Deleted: {file_name}")



# First get working directory and the parent directory
dir_cwd = os.getcwd() 
dir_parent = os.path.dirname(dir_cwd)
dir_tmp = os.path.join(dir_parent, "tmp")
dir_data = os.path.join(dir_parent, "data")
dir_staging = os.path.join(dir_data, "staging")

# Set the paths of inputs and outputs
file_raw_tar = os.path.join(dir_parent, "data","raw","raw.tar.xz")
file_out_tar = os.path.join(dir_parent, "data","staging","compressed_parquet.tar.xz")


os.makedirs(dir_tmp, exist_ok=True)
clean_dir(dir_tmp,"WIPE")

n_total = count_files(file_raw_tar)

for n in range(n_total):
    c_pdf = extract_nth_file(file_raw_tar, dir_tmp, n)
    cr, dt = crop_and_extract(c_pdf, (442, 179, 574, 561))
    c_df = extract_to_df(cr, dt)
    c_df.write_parquet(f'{dir_tmp}/df_{n}.parquet')
    if os.path.exists(c_pdf): 
        os.remove(c_pdf)



###########################################


# Get all Parquet files in the directory
parquet_files = [os.path.join(dir_tmp, f) for f in os.listdir(dir_tmp) if f.endswith(".parquet")]

# Create and compress them into tar.xz
with tarfile.open(file_out_tar, "w:xz") as tar:
    for file in parquet_files:
        tar.add(file, arcname=os.path.basename(file)) 

print(f"Compressed {len(parquet_files)} Parquet files into {file_out_tar}")


df = pl.scan_parquet(f"{dir_tmp}/*.parquet").collect()
df.write_parquet(f'{dir_staging}/fct_parking.parquet')


df2 = pl.DataFrame({
    "Name": ["Surface", "Economy",
             "Level 1","Level 2","Level 3",
             "Level 4","Level 5","Level 6"],
    "Total": [636, 995,
              379,958,954,
              973,320,180],
    "Cost": [8, 6,
             10,10,10,
             10,10,10]
})
df2.write_parquet(f'{dir_staging}/dim_info.parquet')




shutil.rmtree(dir_tmp)