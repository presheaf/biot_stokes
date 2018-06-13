import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle
import itertools

def dezero(blocks):                 # replaces 0-s with np.zeros() of apporpriate size
    N = len(blocks)
    Ns = []
    for i in range(N):
        for block in blocks[i]:
            try:
                Ns.append(block.shape[0])
                break
            except:
                continue
    assert len(Ns) == N
            
    for i in range(N):
        for j in range(N):
            if blocks[i][j] is 0: # == 0 will do eltwise check
                blocks[i][j] = np.zeros((Ns[i], Ns[j]))
    return blocks

def to_numpy(block_mat, block=False):
    """Given a block.block matrix, .arrays() all blocks, inserts np.zeros
    where appropriate and returns it as non-block matrix. If block=True is set,
    leaves block structure in place (still .arrays() and np.zeros everything)"""
    numpy_block_mat = dezero([
        [np.matrix(elt.array()) if elt != 0 else 0
        for elt in row]
        for row in block_mat
    ])

    if not block:
        return np.block(numpy_block_mat)
    else:
        return numpy_block_mat
