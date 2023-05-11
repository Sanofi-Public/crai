import numpy as np
import pymol2
import scipy.spatial.distance
from sklearn.decomposition import PCA


def get_template(pdb_path):
    """
    The goal is to remove asymetric units by comparing their coordinates
    :param pdb_path1:
    :param sel1:
    """
    # pseudoatom origin, pos=[0,0,0]
    # pseudoatom ptx, pos=[10,0,0]
    # distance /origin, /ptx
    # pseudoatom pty, pos=[0,15,0]
    # distance /origin, /pty
    # pseudoatom ptz, pos=[0,0,20]
    # distance /origin, /ptz

    with pymol2.PyMOL() as p:
        p.cmd.load(pdb_path, 'toto')
        sel1 = f'toto  and polymer.protein and (chain H or chain L)'
        p.cmd.extract("sel1", sel1)
        coords1 = p.cmd.get_coords("sel1")
        coords = coords1 - coords1.mean(axis=0)
        pca = PCA(svd_solver='full')
        # fit transform has a negative determinant and breaks chirality
        pca.fit(coords)
        component_matrix = pca.components_
        # I = np.matmul(component_matrix, component_matrix.T)
        # dists = scipy.spatial.distance.pdist(coords)
        # rot_dists = scipy.spatial.distance.pdist(rotated)
        # abs = (np.absolute(dists - rot_dists))
        # np.max(abs)
        determinant = np.linalg.det(component_matrix)
        if determinant < 0:
            component_matrix[-1, :] = -1 * component_matrix[-1, :]
        new_coords = np.matmul(component_matrix, coords.T).T

        # Let's have z as the rotation/main axis
        new_coords = (new_coords[:, [1, 2, 0]])
        p.cmd.load_coords(new_coords, "sel1", state=1)
        # p.cmd.save("aligned.pdb", "sel1", state=1)

        # Now let's crop the Fv and shift the com
        cropped_sel = "sel1 and ((chain H and resi 1-155) or (chain L and resi 1-131))"
        p.cmd.extract("cropped", cropped_sel)
        cropped = p.cmd.get_coords("cropped")
        cropped = cropped - cropped.mean(axis=0)
        p.cmd.load_coords(cropped, "cropped", state=1)
        p.cmd.save("cropped.pdb", "cropped", state=1)

        # select toto, chain H and resi 1-155
        # select titi, chain L and resi 1-131

        # Let's create random dummies
        # from scipy.spatial.transform import Rotation as R
        # r = R.from_rotvec(np.pi / 3 * np.array([0, 0, 1]))
        # rotated = r.apply(new_coords)
        # translated = rotated + np.array([10, 20, 30])[None, :]
        # p.cmd.load_coords(translated, "sel1", state=1)
        # p.cmd.save("rotated.pdb", "sel1", state=1)
        #
        # r = R.from_rotvec(np.pi / 3 * np.array([1, 0, 1]))
        # rotated = r.apply(new_coords)
        # translated = rotated + np.array([10, 20, 30])[None, :]
        # p.cmd.load_coords(translated, "sel1", state=1)
        # p.cmd.save("rotated_2.pdb", "sel1", state=1)


get_template('../data/pdb_em/7LO8_23464/7LO8.cif')
