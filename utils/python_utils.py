import os
import wget


def download_with_overwrite(url, outname, overwrite=False):
    if os.path.exists(outname):
        if overwrite:
            os.remove(outname)
            wget.download(url, out=outname)
    else:
        wget.download(url, out=outname)
