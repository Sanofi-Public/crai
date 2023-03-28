# crIA-EM

```shell
conda create -n cria
conda activate cria
conda install -c schrodinger pymol
pip install wget requests pandas tqdm mrcfile scipy mmtf-python
```

## Listing and downloading systems
First step is to get the antibody density:pdb mapping. To do so, we first get all systems information from SabDab.
Then we removed model with number >0 : just one system had more than one : 7mt{a,b}.
Then we queried the PDB to get corresponding density ids and removed maps for which we have no maps : 1qgc.pdb.
Finally, we downloaded the maps and pdbs in folders named pdbid_emdbid.

## Processing systems
Once we have done this, we want to
