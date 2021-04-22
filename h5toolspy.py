'''
author: John Donaghy
'''

from bs4 import BeautifulSoup
import os
import h5py
import numpy as np
import xarray as xr
from math import ceil
import matplotlib
import matplotlib.pyplot as plt

class H5Processor:
    def __init__(self, root: str, timeAveraged: bool=True):
        self.root = root
        self.baseFile = 'tfd' if timeAveraged else 'pfd'
        
        self.time = 0
        self.xUnits, self.yUnits, self.zUnits = self.__buildGrid()
        self.slice = None
        
        ##temporary for harris
        self.attributes= []
        
    def __nearest(self, array, value, op):
            """
            return index of nearest value 
            """
            if value is not None:
                return np.abs(array - value).argmin()
            else:
                return 0 if op == 'min' else len(array)
        
    def __buildGrid(self):
        
        
        with open(os.path.join(self.root, self.baseFile +'.xdmf'),  'r') as f:
            soup = BeautifulSoup(f, features='html.parser')

        timeFilesXdmf = soup.grid.find_all('xi:include')
        timeFileXdmf = timeFilesXdmf[self.time]['href']

        with open(self.root + timeFileXdmf, 'r') as f1:
            soup1 = BeautifulSoup(f1, features='html.parser')

        dataItems = soup1.find_all('dataitem')
        origin = dataItems[0]

        origin = np.array(origin.contents[0].strip().split()).astype('float')
        DxDyDz = dataItems[1]
        DxDyDz = np.array(DxDyDz.contents[0].strip().split()).astype('float')

        topology = soup1.find_all('topology')[0]
        topology = np.array(topology.attrs['dimensions'].split(" ")).astype('int') - 1
        
        stop = origin + DxDyDz*topology 
        
        zUnits = np.linspace(origin[0], stop[0], topology[0])
        yUnits = np.linspace(origin[1], stop[1], topology[1])
        xUnits = np.linspace(origin[2], stop[2], topology[2])

        return xUnits, yUnits, zUnits
        
    
    def getH5Grid(self,
                  field: str,
                  time: int,
                  xmin: float = None,
                  xmax: float = None,
                  ymin: float = None,
                  ymax: float = None,
                  zmin: float = None,
                  zmax: float = None):
        """
        args:
            field [str]: jx_ec, jy_ec, jz_ec, ex_ec, ey_ec, ez_ec, hx_fc, hy_fc, hz_fc
            time [int]: index into list of saved files of type file_
            (x,y,z)(min,max) [float]: coordinates to slice fields. lazy execution loading.  min==max for 2D slice

        returns:
            slice of grid stored from specified time as a xarray
        """
        
        file_ = self.baseFile if field in ['jx_ec', 'jy_ec', 'jz_ec', 
                                   'ex_ec', 'ey_ec', 'ez_ec', 
                                   'hx_fc', 'hy_fc', 'hz_fc'] else self.baseFile + '_moments'  # Hack for Harris_moments' remove + '_moments'

        if time != self.time:
            self.time = time
        
        with open(os.path.join(self.root, file_ +'.xdmf'),  'r') as f:
            soup = BeautifulSoup(f, features='html.parser')

        timeFilesXdmf = soup.grid.find_all('xi:include')
        timeFileXdmf = timeFilesXdmf[self.time]['href']
        print(f'Loading {field} from File: {timeFileXdmf}')

        self.chkptTime = timeFileXdmf.split('.')[1]#[1:] ##slice off leading 0
        
        with open(self.root + timeFileXdmf, 'r') as f1:
            soup1 = BeautifulSoup(f1, features='html.parser')

        attributeFields = soup1.find_all('attribute')
        
        ## temporary for harris
        if not self.attributes:
            for af in attributeFields:
                self.attributes.append(af.attrs['name'])
        
        for i in range(len(attributeFields)):
            if attributeFields[i].attrs['name'] == field:
                dataFile, keys = attributeFields[i].dataitem.get_text().split()[0].split(':')
                break


        dataFile = h5py.File(os.path.join(self.root, dataFile), 'r')
        key = keys.split('/')[1]

        xmin = self.__nearest(self.xUnits, xmin, 'min')
        xmax = self.__nearest(self.xUnits, xmax, 'max')
        ymin = self.__nearest(self.yUnits, ymin, 'min')
        ymax = self.__nearest(self.yUnits, ymax, 'max')
        zmin = self.__nearest(self.zUnits, zmin, 'min')
        zmax = self.__nearest(self.zUnits, zmax, 'max')

        if zmax == zmin: 
            zmax = zmin + 1
            self.slice = zmin
        elif ymax == ymin:
            ymax = ymin + 1
            self.slice = ymin
        elif xmax == xmin:
            xmax = xmin + 1
            self.slice = xmin

        zUnits = self.zUnits[zmin:zmax]
        yUnits = self.yUnits[ymin:ymax]
        xUnits = self.xUnits[xmin:xmax]

        data = dataFile[key][field]['p0']['3d'][zmin:zmax, ymin:ymax, xmin:xmax]

        grid = xr.DataArray(data, 
                            coords = [zUnits, yUnits, xUnits],
                            dims = ['z', 'y', 'x']
                           )
        return grid  
    
    def selectSubSpace(self, xp: tuple, yp: tuple, axes: tuple=('y','z'), color: str='g', cellsPerPatch: int=32):
        """
        Takes requested subspace limits and converts to integer number of patches most
        nearly representative of requested subspace. Plots selected patches.
        Only supports square subspaces
        used to collect particles from checkpoint file, which are collected patchwise 
        args:
            xp [tuple/int]: (xleft, xright)
            yp [tuple/int]: (ybottom, ytop)
            axis [tuple/str]: (xaxis dimension, yaxis dimension)
            color [str]: matplotlib color
        returns:
            ((int,int,int), int): lower left corner of selected space, number of patches per dim
        """
        xleft, xright = xp
        ybottom, ytop = yp

        dimx, dimy = axes

        if (dimx or dimy) not in ['x', 'y', 'z']:
            raise Exception('Improper spatial coordinates for plot axis')
        slice_ = ['x','y','z']
        slice_.remove(dimx)
        slice_.remove(dimy)
        slicedim = slice_[0]
        
        #convert requested coordinates to cell indexes
        xunits = getattr(self, dimx + 'Units')
        yunits = getattr(self, dimy + 'Units')
        sliceUnits = getattr(self, slicedim + 'Units')
        
        ixl = self.__nearest(xunits, xleft, 'min')
        ixr = self.__nearest(xunits, xright, 'max')
        iyb = self.__nearest(yunits, ybottom, 'min')
        iyt = self.__nearest(yunits, ytop, 'max')

        islice = self.__nearest(sliceUnits, self.slice, 'min')
        #calculate number of patches requested
        span = (ixr - ixl) 
        nPatch = 1 if span/cellsPerPatch < 1 else span//cellsPerPatch
        #patch alignment and covnert back to coordinates
        alignx, aligny = ixl//cellsPerPatch * cellsPerPatch, iyb//cellsPerPatch * cellsPerPatch
        alignSlice = islice//cellsPerPatch * cellsPerPatch
        lLeftX = xunits[alignx]
        lLeftY = yunits[aligny]
        
        idxMap = {'x':0, 'y':1, 'z':2}
        if dimx == 'y' and dimy == 'z':
            cornerPatchIdx = (alignSlice, alignx, aligny)
        elif dimx == 'y' and dimy == 'x':
            cornerPatchIdx = (aligny, alignx, alignSlice)
            
        cornerCoord = (lLeftX, lLeftY)
        
        width = nPatch * cellsPerPatch
        height = nPatch * cellsPerPatch
        print(cornerCoord, width, height)
        rect = matplotlib.patches.Rectangle(cornerCoord, width, height, linewidth=1,edgecolor=color,facecolor='none')
        plt.gca().add_patch(rect)
        return cornerPatchIdx, nPatch

    
    def getPatches(self, origin: tuple, nrows: int, ncols: int, axes: tuple=('y','z'), cellsPerPatch: int=32, spacing: int=1):
        """
        Takes requested subspace limits and converts to integer number of patches most
        nearly representative of requested subspace. Plots selected patches.
        Only supports square subspaces
        used to collect particles from checkpoint file, which are collected patchwise
        grid indexed into from bottom left
        args:
            origin [tuple/float]: (xleft, ybottom) 
            nrows [int]: number of rows
            ncols [int]: number of cols
            axes [tuple/str]: (xaxis dimension, yaxis dimension)
            cellsPerPatch [int]: number of grid cells in patch
            spacing [int]: number of patches between subspace selection
        returns:
            ((int,int,int), int): lower left corner of selected space, number of patches per dim
        """
        xleft, ybottom = origin

        dimx, dimy = axes

        if (dimx or dimy) not in ['x', 'y', 'z']:
            raise Exception('Improper spatial coordinates for plot axis')
        slice_ = ['x','y','z']
        slice_.remove(dimx)
        slice_.remove(dimy)
        slicedim = slice_[0]
        
        #convert requested coordinates to cell indexes
        xunits = getattr(self, dimx + 'Units')
        yunits = getattr(self, dimy + 'Units')
        sliceUnits = getattr(self, slicedim + 'Units')
        
        ixl = self.__nearest(xunits, xleft, 'min')
        iyb = self.__nearest(yunits, ybottom, 'min')

        islice = self.__nearest(sliceUnits, self.slice, 'min')
        #calculate number of patches requested
        span = 1 #fix this
        nPatch = 1 if span/cellsPerPatch < 1 else span//cellsPerPatch
        
        #patch alignment and covnert back to coordinates
        alignx, aligny = ixl//cellsPerPatch * cellsPerPatch, iyb//cellsPerPatch * cellsPerPatch
        alignSlice = islice//cellsPerPatch * cellsPerPatch
        
        cornerPatchIdx = []
        width = xunits[nPatch * cellsPerPatch] - xunits[0]
        height = yunits[nPatch * cellsPerPatch] - yunits[0]
        
        c = np.arange(1,nrows*ncols + 1)
        norm = matplotlib.colors.Normalize(vmin=c.min(), vmax=c.max())
        cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet)
        cmap.set_array([])
        
        for i in range(nrows):
            for j in range(ncols):
                y = aligny + i * (cellsPerPatch * nPatch + cellsPerPatch*spacing)
                x = alignx + j * (cellsPerPatch * nPatch + cellsPerPatch*spacing)

                if dimx == 'y' and dimy == 'z':
                    cornerPatchIdx.append((alignSlice, x, y))
                elif dimx == 'y' and dimy == 'x':
                    cornerPatchIdx.append((y, x, alignSlice))
            
                lLeftX = xunits[x]
                lLeftY = yunits[y]
                cornerCoord = (lLeftX, lLeftY)
                

                rect = matplotlib.patches.Rectangle(cornerCoord, width, height, linewidth=1, edgecolor=cmap.to_rgba(1 + i*ncols + j),facecolor='none')
                plt.gca().add_patch(rect)
                
        return cornerPatchIdx, nPatch
