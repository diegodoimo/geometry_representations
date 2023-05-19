import argparse
import requests
import pathlib
import zipfile

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./results")
    args = parser.parse_args()
    return args


def main(args):
    print('')
    urls = ['https://figshare.com/ndownloader/files/40694132']
    pathlib.Path(f'{args.path}').mkdir(parents=True, exist_ok=True)
    for url in urls:
        r = requests.get(url, stream=True)
        with open(f"{args.path}/data_transf_repr.zip", "wb") as f:
            f.write(r.content)

    with zipfile.ZipFile(f"{args.path}/data_transf_repr.zip", 'r') as zip_ref:
        zip_ref.extractall(f"{args.path}")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)