#!/bin/bash

# URL of the webpage to convert
URL="https://www.msnairport.com/parking_transportation/parking"

# Get the current timestamp
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Output PDF file path
OUTPUT_PATH="/workspace/output/raw/pdf_$TIMESTAMP.pdf"

# Convert webpage to PDF
wkhtmltopdf $URL $OUTPUT_PATH
# Print the output file path
echo "PDF saved as $OUTPUT_PATH"

