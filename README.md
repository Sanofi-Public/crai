# crIA-EM

## Installation

The first thing you will need is a computing environment :

```shell
conda create -n crai -y
conda activate crai
conda install -c schrodinger pymol
#conda install pytorch=1.13 pytorch-cuda=11.7 -c pytorch -c nvidia # For cuda support
conda install pytorch=1.13 cpuonly -c pytorch # For cpu 
pip install wget requests pandas tqdm mrcfile scikit-learn scipy mmtf-python matplotlib tensorboard
```

Then, clone the project and go to the root of the project :
```shell
git clone https://github.com/Vincentx15/crIA-EM.git
cd crIA-EM
```

Finally, one can run predictions on one or several systems by running the command
```shell
conda activate crai
python crai_predict.py -h  # To see a help menu
python crai_predict.py --in PATH_TO_MAP.map  # Single prediction
python crai_predict.py --in PATH_TO_DIR --predict_dir  # Directory prediction
```


## Data processing explanation

### Listing and downloading systems
First step is to get the antibody density:pdb mapping. To do so, we first get all systems information from SabDab.
Then we removed model with number >0 : just one system had more than one : 7mt{a,b}.
Then we queried the PDB to get corresponding density ids and removed maps for which we have no maps : 1qgc.pdb.
Finally, we downloaded the maps and pdbs in folders named pdbid_emdbid.

### Mrc format
The mrcfile package helps to deal with cryo-EM maps, which is a bit convoluted. 
Instead of having the data aligned with the PDB xyz axis, there is an axis mapping tying the data array with the xyz space.
There is a data origin, voxel size (in xyz space) as well as a shift array that is called nxstart but operates in src space.
To standardize those, we introduce an MRCGrid class that is straightforward to work with. (done in mrc_utils.py)
It just has an XYZ array with a single origin in xyz space. 

### Processing systems
Once we have downloaded our raw data, we want to refine our maps. 
They often contain unused regions.
For this reason, we start by creating a carved version that represents the mrc file cut around the PDB file.
Then we resample our maps to have a given resolution for our dataset.
We do this for all the maps in our dataset, so that we have regularly sampled densities around our PDB files.

### Loading our data
Once we have all of our data in this standard way, we are ready to process it, and use it in a learning pipeline.
We load the mrc, then put the corresponding PDB in the same grid and split it into AB/AG/Void channels.



