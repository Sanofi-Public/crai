import pymol.cmd as cmd


def get_protein_coords(pdb_path, pdb_name=None, selection=None, remove_hydrogen=True):
    """
    The goal is to go from pdb files and optionnally some selections to the (n,1)
    Using a name is useful for multiprocessing and not the default
    """
    pdb_name = 'toto' if pdb_name is None else pdb_name

    # Load the protein, prepare the general selection
    cmd.load(pdb_path, pdb_name)
    if remove_hydrogen:
        cmd.remove('hydrogens')
    prot_selection = f'{pdb_name} and polymer.protein'
    selection = prot_selection if selection is None else f"{prot_selection} and ({selection})"
    coords = cmd.get_coords(selection=f'{selection}')
    cmd.delete(pdb_name)
    return coords


'''
# Just to benchmark the parsing time against pymol.
# To use coordinates, pymol is 10 times faster.

from Bio.PDB import MMCIFParser
def biopython_parser(pdbname, parser):
    structure = parser.get_structure("poulet", pdbname)
    coords = [atom.get_vector() for atom in structure.get_atoms()]
    return coords

a = time.perf_counter()
for i in range(10):
    biopython_parser(pdbname, parser=parser)
print(f'time1 : {time.perf_counter() - a}')
# 3.16s vs 0.27 for pymol
'''


def list_id_to_pymol_sel(list_of_ids):
    """
    ['A','B'] => chain A or chain B
    ['A'] => chain A
    :param list_of_ids:
    :return:
    """
    return ' or '.join([f"chain {chain}" for chain in list_of_ids])


def save_coords(coords, topology, outfilename, selection=None):
    """
    Save the coordinates to a pdb file
    • coords: coordinates
    • topology: topology
    • outfilename: name of the oupyt pdb
    • selection: Boolean array to select atoms
    """
    object_name = 'struct_save_coords'
    cmd.delete(object_name)
    if selection is None:
        selection = np.ones(len(topology['resids']), dtype=bool)
    for i, coords_ in enumerate(coords):
        if selection[i]:
            name = topology['names'][i]
            resn = topology['resnames'][i]
            resi = topology['resids'][i]
            chain = topology['chains'][i]
            elem = name[0]
            cmd.pseudoatom(object_name,
                           name=name,
                           resn=resn,
                           resi=resi,
                           chain=chain,
                           elem=elem,
                           hetatm=0,
                           segi=chain,
                           pos=list(coords_))
    cmd.save(outfilename, selection=object_name)
    cmd.delete(object_name)
