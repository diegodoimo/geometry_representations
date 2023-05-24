import argparse
import requests
import pathlib
import zipfile

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="image")
    args = parser.parse_args()
    return args


def main(args):    
    if args.dataset == 'image':
        print('Downloading image dataset')
    
        urls = ['https://figshare.com/ndownloader/files/40694132']
        pathlib.Path(f'{args.path}').mkdir(parents=True, exist_ok=True)
        for url in urls:
            r = requests.get(url, stream=True)
            with open(f"{args.path}/data_transf_repr_image.zip", "wb") as f:
                f.write(r.content)

        with zipfile.ZipFile(f"{args.path}/data_transf_repr_image.zip", 'r') as zip_ref:
            zip_ref.extractall(f"{args.path}")

    elif args.dataset == 'protein':
        print('Downloading protein dataset')

        urls = ['https://figshare.com/ndownloader/articles/23137958/versions/1']
        pathlib.Path(f'{args.path}').mkdir(parents=True, exist_ok=True)
        for url in urls:
            r = requests.get(url, stream=True)
            with open(f"{args.path}/data_transf_repr_protein.zip", "wb") as f:
                f.write(r.content)

        with zipfile.ZipFile(f"{args.path}/data_transf_repr_protein.zip", 'r') as zip_ref:
            pathlib.Path(f"{args.path+'/data'}").mkdir(parents=True, exist_ok=True)
            zip_ref.extractall(f"{args.path+'/data'}")
    
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
