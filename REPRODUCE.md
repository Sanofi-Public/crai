## Data processing

The steps to gather the database are detailed in prepare_database/README.md.
This will equip you with csvs, structures and maps.
The csvs detail all antibodies examples split following our two possible splits (random and sorted).
The aligned structures and maps are found in data/pdb_em.

Then, the way to load those raw datas in pytorch datasets is detailed in the load_data/ folder.

## Training a model

Once the data steps have been carried out, simply run :
```bash
cd learning/
python train_coords.py --sorted -m example_sorted
```

## Validation of the results


### Producing results files

The results files corresponding to dock_in_map are available in the repository data/csvs/ folder.
They can be produced by running :
```bash
cd paper
python benchmark.py
```

To produce results files for a trained model, run :
```bash
python relog.py --nano --sorted --thresh --model_name example_sorted_last
```

This will produce a pickle file containing per systems results.

### Producing figures

Once such files are produced, you can reproduce the figures of the paper by running 
```bash
cd paper
python analyze.py
```

