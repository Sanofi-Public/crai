## Data processing pipeline

### Getting the right pairs and downloading the data
This is done in `download_data.py`
The data is originally fetched from SabDab, and a few obsolete systems are removed : 6mf7 and 7ny5.
We also fix a weird duplicate for 7XOD where chains R and S were considered separate, and remove nanobodies from the 
initial data. 
We remove 7khf whose structure is very weird.
They were found to cause bugs later on.

Then, the PDB ids and chain selections are retrieved, and stored in `cleaned.csv`.
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



