## Data processing pipeline

### Getting the initial data
The data is originally fetched from SabDab.
A few obsolete systems are removed 7ny5.
We remove very weird structures (fusion protein or immunoglobulins M)
7khf, 8blq, 7uvl, 8ema, 8ae{0-1-3}, 8ad{x,y}, 7xq8, 7rxd, 7rxc, 7l6m
They were found to cause bugs later on.

We also fix systems where chains are considered separate :
7YVN, chain HI
7YVP, chain IJ
8HHX, chain HI
8HHY, FW-GI
8DXS, FI-GJ
7XOD, RS-UV-XY
7MLV, KF
6XJA (missing chain AB and missing antigen)

Nanobodies were present in the fab files. They were annotated as having just one chain and no antigen partner.
We moved them to the nanobody file :
7zl{g-h-i-j} chain K bound to L
8hi{i-j-k} chain N bound to L
7xw6 chain N bound to AB
7sk7 chain K bound to C
7sk5 chain E bound to C
7zyi chain K bound to L
7xod chain T-W-Z bound to S-V-Y
6ni2 chain A bound to B | V
7wpe chain W-Z bound to V-Y
6ww2 chain K bound to L
7jhh chain N bound to L
7ul3 chain C bound to A
7tuy chain K bound to L

### Getting the right pairs and downloading the data
This is done in `download_data.py`
The PDB ids and chain selections are retrieved, and stored in `cleaned.csv`.
Then, using the PDB, we find the corresponding cryo-em maps and build a mapping pdb_id : em_id.
We add the mrc column and dump `mapped.csv`
Finally, we download all corresponding maps and cifs.

### Filtering the database
This is done in `filter_database.py`.
Starting from mapped.csv, the parsed output of SabDab, we add missing resolutions by opening the cif files,
yielding the `resolution.csv` file. 
Then, we use phenix.validation_cryoem to get add a validation score column to that file.
Then, we use dock_in_map to try and increase those scores.
We dump `validated.csv` and `docked.csv` files with added columns for both validations
Actually, analysing those results made us realize that these scores were not great even for correct files.
This can be explained by missing B-factors. 
Finally, we process the csv without using validation and docking scores : we simply remove systems with resolution 
below 10A or ones with no antibodies or antigen chains, and group pdb_ids together.
We dump `filtered.csv` and split it to obtain `filtered_{train,val,test}.csv`.

### Processing the database
Once equipped with those splits we are ready to do machine learning. 
However, the map files obtained from the PDB can be enormous (up do 1000³ grid cells) and not centered for viruses :
the pdb only occupies a fraction of the map.
To deal with this we replace the original maps with ones centered around the PDB with a margin of 25A, resampled with a 
voxel size of 2 in files named `f"full_crop_resampled_2.mrc"`.
We also provide cropping around the antibodies and their antigen to get even smaller maps for learning.
To do so, we filter out redundant copies of antibodies (for symmetrical systems) and dump cropped mrc files, along with
a csv keeping track of those systems `chunked_{train,val,test}.csv`.

### Template management
The last thing we need for object detection is an antibody template that serves to transform an antibody into a 
translation and rotation (using Pymol align). 
To get this template, we pick a random Fab system and manually select the Fv residues.
Then we shift the system so that the Fv is centered at the origin and align its main axis with uz vector using PCA.
Given an antibody, pymol align now gives us a translation and rotation to transform our template into it.

## Nanobodies
All of these steps apply to the production of nanobody data. 
We start with the .tsv result of cryo-EM systems containing nanobodies
We manually curate it, as several lines pertain to Fabs instead of nanobodies.
