import os
import hashlib
import wget


def init(l):
    global lock
    lock = l


def download_with_overwrite(url, outname, overwrite=False):
    if outname is not None and os.path.exists(outname):
        if overwrite:
            os.remove(outname)
            wget.download(url, out=outname)
    else:
        wget.download(url, out=outname)


def mini_hash(s):
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 100
