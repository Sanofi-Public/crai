# CrAI

This is the accompanying code for the paper "Finding Antibodies in Cryo-EM densities with CrAI" by Vincent Mallet, Chiara Rapisarda, Hervé Minoux and Maks Ovsjanikov.
The goal of this tool is to predict the position and orientation of antibodies in Cryo-EM densities.

## Installation

### ChimeraX

The easiest way to use the tool is through ChimeraX, as our tool is packaged as a ChimeraX bundle.
The Chimerax toolshed is a [hosted repository](https://cxtoolshed.rbvi.ucsf.edu/) of bundle wheels.
To install the tool, run the ChimeraX application and click More Tools... in the Tools menu.
In the popup, search for "crai" and click the "install" button.
You now should be able to use the tool !

**if it did not work**, you can try downloading the bundle and installing it separately.
You can find the link to the bundle [here](https://cxtoolshed.rbvi.ucsf.edu/apps/chimeraxcrai):
Once downloaded, type the following command in the system command line to install it :
```shell
# system command line
chimerax --nogui --cmd "toolshed install ChimeraX_crai-0.1-py3-none-any.whl; exit"
```

The tool should now be installed in Chimerax.
Once installed, the tool can be used from the Chimerax Command line. 
Examples below include :
```shell
# Chimerax command line
help crai
crai #1 outname YOURNAME.pdb
```

### Command line usage

The first thing you will need is a computing environment :

```shell
conda create -n crai -y
conda activate crai
conda install -c schrodinger pymol
#conda install pytorch=1.13 pytorch-cuda=11.7 -c pytorch -c nvidia # For cuda support
conda install pytorch=1.13 cpuonly -c pytorch # For cpu 
pip install wget requests pandas tqdm mrcfile scikit-learn scipy mmtf-python matplotlib tensorboard cripser
```

Then, clone the project and go to the root of the project :
```shell
git clone https://github.com/Sanofi-GitHub/crai
cd crIA-EM
```

Finally, one can run predictions on one or several systems by running the command
```shell
conda activate crai
python crai_predict.py -h  # To see a help menu
python crai_predict.py --in PATH_TO_MAP.map  # Single prediction
python crai_predict.py --in PATH_TO_DIR --predict_dir  # Directory prediction
```

## Contact

If you encounter difficulties installing or using the tool, please open an issue on GitHub or contact me directly
at the following email address : vincent.mallet96 at gmail.com
