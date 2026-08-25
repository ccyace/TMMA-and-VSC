import os
import hashlib
import time
import requests
from tqdm import tqdm

URL_MAP = {
    "cifar10": "https://heibox.uni-heidelberg.de/f/869980b53bf5416c8a28/?dl=1",
    "ema_cifar10": "https://heibox.uni-heidelberg.de/f/2e4f01e2d9ee49bab1d5/?dl=1",
    "lsun_bedroom": "https://heibox.uni-heidelberg.de/f/f179d4f21ebc4d43bbfe/?dl=1",
    "ema_lsun_bedroom": "https://heibox.uni-heidelberg.de/f/b95206528f384185889b/?dl=1",
    "lsun_cat": "https://heibox.uni-heidelberg.de/f/fac870bd988348eab88e/?dl=1",
    "ema_lsun_cat": "https://heibox.uni-heidelberg.de/f/0701aac3aa69457bbe34/?dl=1",
    "lsun_church": "https://heibox.uni-heidelberg.de/f/2711a6f712e34b06b9d8/?dl=1",
    "ema_lsun_church": "https://heibox.uni-heidelberg.de/f/44ccb50ef3c6436db52e/?dl=1",
}
CKPT_MAP = {
    "cifar10": "diffusion_cifar10_model/model-790000.ckpt",
    "ema_cifar10": "ema_diffusion_cifar10_model/model-790000.ckpt",
    "lsun_bedroom": "diffusion_lsun_bedroom_model/model-2388000.ckpt",
    "ema_lsun_bedroom": "ema_diffusion_lsun_bedroom_model/model-2388000.ckpt",
    "lsun_cat": "diffusion_lsun_cat_model/model-1761000.ckpt",
    "ema_lsun_cat": "ema_diffusion_lsun_cat_model/model-1761000.ckpt",
    "lsun_church": "diffusion_lsun_church_model/model-4432000.ckpt",
    "ema_lsun_church": "ema_diffusion_lsun_church_model/model-4432000.ckpt",
}
MD5_MAP = {
    "cifar10": "82ed3067fd1002f5cf4c339fb80c4669",
    "ema_cifar10": "1fa350b952534ae442b1d5235cce5cd3",
    "lsun_bedroom": "f70280ac0e08b8e696f42cb8e948ff1c",
    "ema_lsun_bedroom": "1921fa46b66a3665e450e42f36c2720f",
    "lsun_cat": "bbee0e7c3d7abfb6e2539eaf2fb9987b",
    "ema_lsun_cat": "646f23f4821f2459b8bafc57fd824558",
    "lsun_church": "eb619b8a5ab95ef80f94ce8a5488dae3",
    "ema_lsun_church": "fdc68a23938c2397caba4a260bc2445f",
}


def _try_remove(path, retries=3, sleep_s=1.0):
    if not os.path.exists(path):
        return True
    for attempt in range(retries):
        try:
            os.remove(path)
            return True
        except PermissionError:
            if attempt + 1 < retries:
                time.sleep(sleep_s)
            else:
                return False
        except OSError:
            return False
    return False


def download_to_part(url, part_path, chunk_size=1024 * 1024):
    os.makedirs(os.path.split(part_path)[0], exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(part_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(len(data))


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def _promote_part_to_final(part_path, final_path):
    if _try_remove(final_path):
        os.replace(part_path, final_path)
        return final_path
    print(
        "Warning: cannot replace locked checkpoint {}. "
        "Close other Python jobs that may be downloading it, then retry.".format(
            final_path
        )
    )
    print("Using verified partial file: {}".format(part_path))
    return part_path


def get_ckpt_path(name, root=None, check=False):
    if 'church_outdoor' in name:
        name = name.replace('church_outdoor', 'church')
    assert name in URL_MAP
    cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    root = (
        root
        if root is not None
        else os.path.join(cachedir, "diffusion_models_converted")
    )
    path = os.path.join(root, CKPT_MAP[name])
    part_path = path + ".part"
    expected_md5 = MD5_MAP[name]

    if os.path.exists(path):
        try:
            if md5_hash(path) == expected_md5:
                return path
        except OSError:
            pass
        print(
            "Checkpoint MD5 mismatch (incomplete download?): expected {}".format(
                expected_md5
            )
        )
        _try_remove(path)

    if os.path.exists(part_path):
        try:
            if md5_hash(part_path) == expected_md5:
                return _promote_part_to_final(part_path, path)
        except OSError:
            pass
        _try_remove(part_path)

    print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
    download_to_part(URL_MAP[name], part_path)
    md5 = md5_hash(part_path)
    assert md5 == expected_md5, md5
    return _promote_part_to_final(part_path, path)
