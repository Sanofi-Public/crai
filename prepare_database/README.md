## Data processing pipeline

### Getting the right pairs and downloading the data

This is done in `download_data.py`
The data is originally fetched from SabDab, where the PDB ids and chain selections are retrieved.
Then, using the PDB, we find the corresponding cryo-em maps and build a mapping pdb_id : em_id.
Finally, we download all corresponding maps and cifs.

### Filtering the database

This is done in `filter_database.py`
Starting from cleaned.csv, the parsed output of SabDab, we add missing resolutions by opening the cif files,
yielding the cleaned_res.csv file. 
Then, we use phenix.validation_cryoem to get add a validation score column to that file.
Then, we use dock_in_map to try and increase those scores.
Actually, analysing those results made us realize that these scores were not great even for correct files.
This can be explained by missing B-factors. 
Finally, we process the csv without using validation and docking scores : we simply remove systems with resolution 
below 10A or ones with no antibodies or antigen chains, and group pdb_ids together.

### Processing the database


