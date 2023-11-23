import requests
import zipfile
import os
from tqdm import tqdm

# URL of the dataset
url = "https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/slovo/slovo.zip"

# The directory to save the dataset
directory = "data"

# Create the directory if it does not exist
if not os.path.exists(directory):
    os.makedirs(directory)
    print("Directory created")

# Path to save the zip file
zip_path = os.path.join(directory, "slovo.zip")

# Downloading the dataset with a progress bar
response = requests.get(url, stream=True)
total_size_in_bytes = int(response.headers.get('content-length', 0))
block_size = 1024 # 1 Kibibyte
progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

with open(zip_path, "wb") as file:
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        file.write(data)
progress_bar.close()

if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
    print("ERROR, something went wrong")

# Unzipping the dataset
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(directory)

# Removing the zip file after extraction
os.remove(zip_path)

print("Dataset downloaded and extracted to:", directory)