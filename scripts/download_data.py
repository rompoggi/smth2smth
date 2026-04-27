#/usr/bin/python3
"""Script to download the something 2 something dataset"""

import gdown
import zipfile
import os

if __name__ == "__main__":
    file_id = '1SlRJBD6cyXMr5772kOKe5xXAU9Scu5vR'
    url = f'https://drive.google.com/uc?id={file_id}'
    output_zip = 'downloaded_file.zip'
    extract_to = './data'

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        print(f"Created directory: {extract_to}")

    print("Downloading from Google Drive...")
    gdown.download(url, output_zip, quiet=False)

    print(f"Extracting to {extract_to}...")
    try:
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete!")
        
        os.remove(output_zip)
        print(f"Removed {output_zip}")

    except zipfile.BadZipFile:
        print("Error Extracing Zip")
