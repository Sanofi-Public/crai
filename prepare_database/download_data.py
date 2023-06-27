import os
import sys

import numpy as np
import pandas as pd
import pickle
import requests
from tqdm import tqdm

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from utils.python_utils import download_with_overwrite


def get_ab_list(in_tsv='../data/20230315_0733035_summary.tsv', out_csv='../data/cleaned.csv'):
    """
    Parse SabDab tsv file to get pdbs and infos
    :return:
    """
    df = pd.read_csv(in_tsv, sep='\t')

    # All systems are model=0 except 7ma that has 6 copies
    # models = df[['model']]
    # un = np.unique(models, return_counts=True)
    df = df.loc[df['model'] == 0]
    df = df[['pdb', 'Hchain', 'Lchain', 'antigen_chain', 'resolution']]
    df.to_csv(out_csv)
    return df


def get_mapping_ids(list_of_ids=("6GH5", "3JAU")):
    """
    PDB query to get the EMDB numbers
    """
    # One need to format this as a GQL query, hence turn it all in a big string
    list_as_string = '["' + '","'.join(list_of_ids) + '"]'
    gql_query = '{entries(entry_ids:' + list_as_string + '){rcsb_id,rcsb_entry_container_identifiers{emdb_ids}}}'
    url_query = f'https://data.rcsb.org/graphql?query={gql_query}'
    r = requests.get(url_query)
    json_result = r.json()
    mapping_ids = {}
    # counts = []
    for hit_dict in json_result['data']['entries']:
        pdb_id = hit_dict['rcsb_id']
        # This could be a None or a list, but it's actually only ever one thing. (snippet commented)
        # just one empty map for 1qgc, all the rest have just one emdb id
        emdb_ids = hit_dict['rcsb_entry_container_identifiers']['emdb_ids']
        if emdb_ids is None:
            continue
        mapping_ids[pdb_id] = emdb_ids[0]
    #     try:
    #         counts.append(len(emdb_ids))
    #     except:
    #         counts.append(0)
    # un = np.unique(counts, return_counts=True)
    pickle.dump(mapping_ids, open('result_mapping.p', 'wb'))
    return mapping_ids


def add_mrc(csv_pdb, pdb_em_mapping, out_csv):
    pdb_df = pd.read_csv(csv_pdb)
    pdb_df['mrc'] = pdb_df.apply(lambda x: pdb_em_mapping.get(x['pdb'].upper(), '0000missing')[4:], axis=1)
    # for manual inspection : pdb_df.loc[pdb_df['mrc'] == 'missing']
    # We see about 5 obsolete PDBs, that can be fixed in the original tsv by changing the numbers
    pdb_df = pdb_df.loc[pdb_df['mrc'] != 'missing']
    pdb_df.to_csv(out_csv)


def download_one_mrc(emd_id, outdir, overwrite=False):
    header_ftp = f'https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emd_id}/header/emd-{emd_id}.xml'
    map_ftp = f'https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emd_id}/map/emd_{emd_id}.map.gz'
    header_outname = os.path.join(outdir, f'emd-{emd_id}.xml')
    map_outname = os.path.join(outdir, f'emd_{emd_id}.map.gz')

    download_with_overwrite(url=header_ftp, outname=header_outname, overwrite=overwrite)
    download_with_overwrite(url=map_ftp, outname=map_outname, overwrite=overwrite)


def download_one_mmtf(pdb_id, outdir, overwrite=False):
    mmtf_url = f"https://mmtf.rcsb.org/v1.0/full/{pdb_id}.mmtf.gz"
    mmtf_outname = os.path.join(outdir, f'{pdb_id}.mmtf.gz')
    download_with_overwrite(url=mmtf_url, outname=mmtf_outname, overwrite=overwrite)


def download_one_cif(pdb_id, outdir, overwrite=False):
    cif_url = f'http://www.pdb.org/pdb/download/downloadFile.do?fileFormat=cif&structureId={pdb_id}'
    cif_outname = os.path.join(outdir, f'{pdb_id}.cif')
    download_with_overwrite(url=cif_url, outname=cif_outname, overwrite=overwrite)


def get_database(mapping, root='../data/pdb_em', overwrite=False):
    """
    Builds a database with flat files

    :param mapping: The output of get_mapping_ids
    :param root: Where to build the dataset
    :return:
    """
    for pdb, em in tqdm(sorted(mapping.items())):
        # 1YCR + EMD-123 -> 1YCR_123
        em_id = em[4:]
        dir_to_build = os.path.join(root, f'{pdb}_{em_id}')
        os.makedirs(dir_to_build, exist_ok=True)
        # download_one_mmtf(pdb_id=pdb, outdir=dir_to_build, overwrite=overwrite)
        download_one_cif(pdb_id=pdb, outdir=dir_to_build, overwrite=overwrite)
        download_one_mrc(emd_id=em_id, outdir=dir_to_build, overwrite=overwrite)


if __name__ == '__main__':
    max_systems = None
    csv_pdb = '../data/cleaned.csv'
    pdb_df = get_ab_list(out_csv=csv_pdb)
    # pdb_df = pd.read_csv(csv_pdb)
    relevant_ids = np.unique(pdb_df['pdb'])[:max_systems]
    pdb_em_mapping = get_mapping_ids(relevant_ids)

    csv_mapped = '../data/mapped.csv'
    add_mrc(csv_pdb=csv_pdb, pdb_em_mapping=pdb_em_mapping, out_csv=csv_mapped)
    # download_one_mrc()
    # download_one_mmtf()
    get_database(pdb_em_mapping)

    # path = '../data/pdb_em'
    # for system in os.listdir(path):
    #     print(system)
    #     pdb_name, mrc_name = system.split("_")
    #     outdir = os.path.join(path, system)
    #     download_one_cif(pdb_id=pdb_name, outdir=outdir, overwrite=False)
