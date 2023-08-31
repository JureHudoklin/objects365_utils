import os
import torch
import tqdm
import signal

from multiprocessing import Pool
from pathlib import Path
from tarfile import is_tarfile
from zipfile import ZipFile, is_zipfile
from itertools import repeat
from functools import partial

def unzip_file(file, path=None, exclude=(".DS_Store", "__MACOSX")):
    # Unzip a *.zip file to path/, excluding files containing strings in exclude list
    if path is None:
        path = Path(file).parent  # default path
    with ZipFile(file) as zipObj:
        for f in zipObj.namelist():  # list all archived filenames in the zip
            if all(x not in f for x in exclude):
                zipObj.extract(f, path=path)


def download_one(
    url_dir, retry=1, curl=False, unzip=True, delete_zip=True, show_progress=True
):
    url = url_dir[0]
    dir = url_dir[1]
    # Download 1 file
    success = True

    # Check if Path(url).name already exists
    f_name = Path(url).name
    f_name = f_name.replace(".tar.gz", "")
    if f_name in os.listdir(dir):
        print(f"{f_name} exists, skipping download...")
        return

    # Check if file is downloaded but not unzipped
    f = dir / Path(url).name
    if f.is_file():
        print(f"{f} exists, skipping download...")
    else:
        f = dir / Path(url).name
        print(f"Downloading {url} to {f}...")
        for i in range(retry + 1):
            # Attempt file download
            if curl:
                s = "" if show_progress > 1 else "sS"  # silent
                r = os.system(
                    f'curl -# -{s}L "{url}" -o "{f}" --retry 9 -C -'
                )  # curl download with retry, continue
                success = r == 0
            else:
                torch.hub.download_url_to_file(
                    url, f, progress=show_progress
                )  # torch download
                success = f.is_file()

            # Check if download was successful
            if success:
                break
            elif i < retry:
                print(f"⚠️ Download failure, retrying {i + 1}/{retry} {url}...")
            else:
                print(f"❌ Failed to download {url}...")

    if unzip and success and (f.suffix == ".gz" or is_zipfile(f) or is_tarfile(f)):
        print(f"Unzipping {f}...")
        try:
            if is_zipfile(f):
                unzip_file(f, dir)  # unzip
            elif is_tarfile(f):
                os.system(f"tar xf {f} --directory {f.parent}")  # unzip
            elif f.suffix == ".gz":
                os.system(f"tar xfz {f} --directory {f.parent}")  # unzip
            print(f"Unzipping finished {f}...")
            if delete_zip:
                f.unlink()
        except:
            print(f"\n !!!!! ❌ Failed to unzip {f}... !!!!! \n")

def init_worker():
    """
    Catch Ctrl+C signal to termiante workers
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def download(
    urls, dir=".", unzip=True, delete=True, curl=False, num_workers=3, retry=3
):
    # Multiprocessed file download and unzip function, used in data.yaml for autodownload
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory

    p_download_one = partial(
        download_one,
        retry=retry,
        curl=curl,
        unzip=unzip,
        delete_zip=delete,
        show_progress=False,
    )

    print("--- Start Downloading ---")
    p = Pool(num_workers, init_worker)
    try:
        out = []
        for ann in tqdm.tqdm(
            p.imap_unordered(p_download_one, zip(urls, repeat(dir))), total=len(urls)
        ):
            out.append(ann)
    except KeyboardInterrupt:
        print("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()


if __name__ == "__main__":
    dir = Path("object365")  #
    # Make Directories
    for p in "images", "labels":
        (dir / p).mkdir(parents=True, exist_ok=True)
        for q in "train", "val":
            (dir / p / q).mkdir(parents=True, exist_ok=True)

    base_url = "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86"

    train_downloads = [f"{base_url}/train/patch{i}.tar.gz" for i in range(50 + 1)]
    train_downloads.append(f"{base_url}/train/zhiyuan_objv2_train.tar.gz")

    val_downloads = [f"{base_url}/val/images/v1/patch{i}.tar.gz" for i in range(15 + 1)]
    val_downloads.extend(
        [f"{base_url}/val/images/v2/patch{i}.tar.gz" for i in range(16, 43 + 1)]
    )
    val_downloads.append(f"{base_url}/val/zhiyuan_objv2_val.json")

    val_dir = dir / "images" / "val"
    download(val_downloads, dir=val_dir, num_workers=6)

    train_dir = dir / "images" / "train"
    download(train_downloads, dir=train_dir, num_workers=6)
    exit()

