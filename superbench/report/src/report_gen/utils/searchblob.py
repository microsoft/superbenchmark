# https://aimiciusdata.blob.core.windows.net/datastore/raw/CCF/pdf_files/
# https://aimiciusdata.blob.core.windows.net/datastore/raw/CCF/websrc/pdf/
# https://aimiciusdata.blob.core.windows.net/datastore/raw/arxiv/pdf/


import os
import shutil
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.storage.blob import BlobClient

def list_blobs_in_folder(bloburl, container, folder):
    # create the BlobServiceClient object
    default_credential = DefaultAzureCredential()

    blob_service_client = BlobServiceClient(bloburl, credential=default_credential)

    # specify the container name
    container_name = container

    # get the container client
    container_client = blob_service_client.get_container_client(container_name)

    # list the blobs in the container
    print("\nListing blobs...")
    blob_list = container_client.list_blobs(name_starts_with=folder)
    for blob in blob_list:
        print("\t" + blob.name)

def save_file_to_blob(bloburl, container, filepath, local_file):
    # create the BlobServiceClient object
    default_credential = DefaultAzureCredential()

    blob_service_client = BlobServiceClient(bloburl, credential=default_credential)

    # get the blob client
    blob_client = blob_service_client.get_blob_client(container, filepath)

    # upload the local file
    with open(local_file, "rb") as data:
        blob_client.upload_blob(data)

def read_file_from_blob(bloburl, container, filepath):
    # create the BlobServiceClient object
    default_credential = DefaultAzureCredential()

    blob_service_client = BlobServiceClient(bloburl, credential=default_credential)

    # get the blob client
    blob_client = blob_service_client.get_blob_client(container, filepath)

    # download the blob to a local file
    download_stream = blob_client.download_blob()

    # return the content of the file
    return download_stream.readall()

def create_parent_dir(file_path):
    parent = os.path.dirname(file_path)
    if not os.path.exists(parent):
        os.makedirs(parent)
    print(parent)


def download_all_from_blob(bloburl, container, remote_dir, local_dir):
    # create the BlobServiceClient object
    default_credential = DefaultAzureCredential()

    blob_service_client = BlobServiceClient(bloburl, credential=default_credential)

    # get the container client
    container_client = blob_service_client.get_container_client(container)

    # list the blobs in the container
    blob_list = container_client.list_blobs(name_starts_with=remote_dir)

    for blob in blob_list:
        if blob.name.endswith('.json'):
            # print(f"blob.name: {blob.name}")
            blob_client = blob_service_client.get_blob_client(container, blob.name)
            download_file_path = os.path.join(local_dir, blob.name)
            # print(f"download_file_path: {download_file_path}")
            create_parent_dir(download_file_path)
            with open(file=os.path.join(download_file_path), mode="wb") as sample_blob:
                download_stream = blob_client.download_blob()
                sample_blob.write(download_stream.readall())
            # print(f"Blob {blob.name} downloaded to {download_file_path}")

    print(f"All files in {remote_dir} have been downloaded to {local_dir}")
        
def main():
    # step 0, switch to agent venv
    # step 1, az login
    
    # OP1
    # sub = Data-Centric-AI
    # account = sc-xu85
    bloburl = "https://aimiciusdata.blob.core.windows.net"
    container = "datastore"
    folder = "raw/CCF/pdf_files/"
    
    # OP2
    # sub = Shuguang Liu's Team Dev/Test
    # account = lequ
    bloburl = "https://luciastore.blob.core.windows.net/"
    container = "mount"
    folder = "agents/infrawise"
    
    # list
    list_blobs_in_folder(bloburl, container, folder)
    # save
    savefilepath = "agents/infrawise/reportgen/prompt/tmp.md"
    local_file = "/home/lequ/tmp/lucia/infrawise/reportgen/prompt/tmp.md"
    save_file_to_blob(bloburl, container, savefilepath, local_file)
    # list
    #list_blobs_in_folder(bloburl, container, folder)
    # read
    readfilepath = "agents/reliaguard-validation/results_analysis_demo_inputs/all_nodes_standard_nd96asr_v4_BLZ22PrdGPC07.jsonl"
    #content = read_file_from_blob(bloburl, container, readfilepath)
    #print(content)

if __name__ == "__main__":
    main()