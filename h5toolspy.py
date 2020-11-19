'''
author: John Donaghy
'''

from bs4 import BeautifulSoup
import os
import h5py
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt

def getH5Grid(root: str, file_: str, field: str, time: int):
    """
    args:
        root [str]: path to root directory
        file_ [str]: xdmf file to open, usually tfd or tfd_moments
        field [str]: jx_ec, jy_ec, jz_ec, ex_ec, ey_ec, ez_ec, hx_fc, hy_fc, hz_fc
        time [int]: index into list of saved files of type file_

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

    dataItems = soup1.find_all('dataitem')
    origin = dataItems[0]
    
    origin = np.array(origin.contents[0].strip().split()).astype('float')
    DxDyDz = dataItems[1]
    DxDyDz = np.array(DxDyDz.contents[0].strip().split()).astype('float')

    topology = soup1.find_all('topology')[0]
    topology = np.array(topology.attrs['dimensions'].split(" ")).astype('int')  - 1
    

    attributeFields = soup1.find_all('attribute')
    
    for i in range(len(attributeFields)):
        if attributeFields[i].attrs['name'] == field:
            break
            
    dataFile, keys = attributeFields[i].dataitem.get_text().split()[0].split(':')

    dataFile = h5py.File(os.path.join(root, dataFile), 'r')
    key = keys.split('/')[1]
    
    data = np.array(dataFile[key][field]['p0']['3d'])
    stop = origin + DxDyDz*topology

    zUnits = np.linspace(origin[0], stop[0], topology[0])
    yUnits = np.linspace(origin[1], stop[1], topology[1])
    xUnits = np.linspace(origin[2], stop[2], topology[2])
    
    grid = xr.DataArray(data, 
                        coords = [zUnits, yUnits, xUnits],
                        dims = ['z', 'y', 'x']
                       )
    
    return grid
    
def selectSubSpace(xp: tuple, yp: tuple, color: str='g' ):
    """
    args:
        xp [tuple/int]: (xleft, xright)
        yp [tuple/int]: (ybottom, ytop)
        color [str]: matplotlib color 
    """
    xleft, xright = xp
    ybottom, ytop = yp
    width = xright - xleft
    height = ytop - ybottom
    rect = matplotlib.patches.Rectangle((xleft,ybottom), width, height, linewidth=1,edgecolor=color,facecolor='none')
    plt.gca().add_patch(rect)
    return 


