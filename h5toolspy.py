'''
author: John Donaghy
'''

from bs4 import BeautifulSoup
import os
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def getH5Grid(root: str, file_: str, field: str, time: int, corner: tuple, dims: tuple):
    """
    args:
        root [str]: path to root directory
        file_ [str]: xdmf file to open, usually tfd or tfd_moments
        field [str]: jx_ec, jy_ec, jz_ec, ex_ec, ey_ec, ez_ec, hx_fc, hy_fc, hz_fc
        time [int]: index into list of saved files of type file_
        corner [tuple/int]: tuple of ints, lower left hand corner of grid to load
        dims: [tuple/int]: dimensions of grid to load
    returns:
        slice of grid stored in file_ at specified time as a numpy array
    """
    with open(os.path.join(root, file_ +'.xdmf'),  'r') as f:
        soup = BeautifulSoup(f, features='html.parser')

    timeFilesXdmf = soup.grid.find_all('xi:include')
    timeFileXdmf = timeFilesXdmf[time]['href']
    print(f'Loading {field} from File: {timeFileXdmf}')
   
    with open(root + timeFileXdmf, 'r') as f1:
        soup1 = BeautifulSoup(f1, features='html.parser')
        
    attributeFields = soup1.find_all('attribute')
    
    for i in range(len(attributeFields)):
        if attributeFields[i].attrs['name'] == field:
            break
            
    dataFile, keys = attributeFields[i].dataitem.get_text().split()[0].split(':')

    dataFile = h5py.File(os.path.join(root, dataFile), 'r')
    key = keys.split('/')[1]
    
    x, y, z = corner
    dimx, dimy, dimz = dims
    return np.array(dataFile[key][field]['p0']['3d'][z:z+dimz, y:y+dimy, x:x+dimx])
    
def selectSubSpace(xp: int, yp: int, cellsPerPatch = 32, numPatches = 1):
    """
    args:
        yp [int]: Left edge of box
        zp [int]: bottom edge of box
        cellsPerPatch [int]: number of grid cells in a singled dimension of patch
        numPatches [int]: number of patches to select in a single dimension
    """
    width = (cellsPerPatch-1)*numPatches
    rect = matplotlib.patches.Rectangle((xp,yp), width, width, linewidth=1,edgecolor='g',facecolor='none')
    plt.gca().add_patch(rect)
    return 


