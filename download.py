import argparse
import requests
import pathlib
import zipfile
import os


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, default="./results")
    args = parser.parse_args()
    return args


def download_and_extract_dataset(url, filename, target_dir):

    if filename.split(".")[0].split("_")[-1] == "protein":
        target_dir += "/data"
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

    # download the zip file
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{target_dir}/{filename}", "wb") as f:
        r = requests.get(url, stream=True)
        f.write(r.content)

    # extract the zipped content
    with zipfile.ZipFile(f"{target_dir}/{filename}", "r") as zip_ref:
        zip_ref.extractall(target_dir)

    os.remove(f"{target_dir}/{filename}")


if __name__ == "__main__":
    args = parse_arguments()

    datasets = [
        {
            "url": "https://figshare.com/ndownloader/files/40694132",
            "filename": "data_transf_repr_image.zip",
        },
        {
            "url": "https://figshare.com/ndownloader/articles/23137958/versions/1",
            "filename": "data_transf_repr_protein.zip",
        },
    ]

    for dataset in datasets:
        print(f"Downloading {dataset['filename'].split('.')[0]} dataset")
        download_and_extract_dataset(
            dataset["url"], dataset["filename"], args.target_dir
        )

    print(f"\nID AND OVERLAP DATA SAVED IN {args.target_dir}/data")
