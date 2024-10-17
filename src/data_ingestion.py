#src/data_ingestion.py

import os
import requests
from zipfile import ZipFile
import pandas as pd

def download_and_unzip(url, extract_to, output_filename):
    local_zip_path = os.path.join(extract_to, 'data.zip')
    os.makedirs(extract_to, exist_ok=True)

    # Download the zip file
    print("Downloading data...")
    response = requests.get(url)
    with open(local_zip_path, 'wb') as f:
        f.write(response.content)
    print("Download complete.")

    # Extract the zip file
    with ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

    # Assuming the CSV file has the same name as the zip file or you know the name
    extracted_csv_path = os.path.join(extract_to, output_filename)
    if not os.path.exists(extracted_csv_path):
        raise FileNotFoundError(f"Expected file not found in the extracted data: {output_filename}")

    return extracted_csv_path

def save_data_to_csv(data, save_path, file_name):
    final_path = os.path.join(save_path, file_name)
    os.makedirs(save_path, exist_ok=True)
    data.to_csv(final_path, index=False)
    print(f"Data saved in {final_path}")

if __name__ == "__main__":
    url = "https://drive.google.com/uc?export=download&id=14lB0vqMFkaZj8HZ44b0njYgxs9nAN8KO"
    extract_to = '../data/raw'
    output_filename = 'datatran2024.csv'

    extracted_csv_path = download_and_unzip(url, extract_to, output_filename)
    data = pd.read_csv(extracted_csv_path, sep=';', encoding='latin1')  # Specify encoding here
    save_data_to_csv(data, '../data/raw', output_filename)
