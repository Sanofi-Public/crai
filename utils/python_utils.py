import os
import wget


def download_with_overwrite(url, outname, overwrite=False):
    if outname is not None and os.path.exists(outname):
        if overwrite:
            os.remove(outname)
            wget.download(url, out=outname)
    else:
        wget.download(url, out=outname)
