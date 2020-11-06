'''
author: John Donaghy
'''


from scipy import stats
from scipy.interpolate import griddata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import adios2




class Particles:
    """
    Class to collect metadata from PSC checkpoint outputs and provide functionality to JIT load relevant particles
    """
    def __init__(self, path, species='all'):
        """
        args:
            path [str]: path to checkpoint file
            species [str]: electron_he, electron, ion, all
        """
        self.path = path
        self.species = {'electron_he': 0.0, 'electron': 1.0, 'ion': 2.0, 'all': -1}[species]
        self.columns = self.__getColumns()
        self.patchSizes = self.__getSizes()
        self.patchCoordinates = self.__getPatchOffsets()
        self.xb = self.__getXb()
        self.gridMaxDims = self.patchCoordinates[-1]
        
    def __getColumns(self, printVar=False):
        """
        Collect columns from checkpoint file, optionally print.  Columns describe particle fields
        """
        columns = []
        with adios2.open(self.path, 'r') as fh:
            for fstep in fh:
                # inspect variables in current step
                step_vars= fstep.available_variables()
                
                # print variables information
                for name, info in step_vars.items():
                    columns.append(name)
                    if(printVar):
                        print("variable name: " + name)
                        for key, value in info.items():
                            print("\t" + key + ": " + value)
                        print("\n")
        return columns

    def __getSizes(self):
        """
        Collect the number of particles assigned to each patch.  Metadata for entire grid
        """
        if('mprts::mprts::size_by_patch' not in self.columns):
            return 0
        with adios2.open(self.path, 'r') as fh:
            size_by_patch = fh.read('mprts::mprts::size_by_patch')
        return size_by_patch
    
    def __getPatchOffsets(self):
        """
        Collect the global position offsets of patches. Metadata for entire grid
        """
        with adios2.open(self.path, 'r') as fh:
            patchCoordinates = fh.read('grid::off')
        return patchCoordinates
    
    def __getXb(self):
        """
        Collect the offsets of lower left corner of patch.  Offset in number of grid points.  Metadata for entire grid
        """
        with adios2.open(self.path, 'r') as fh:
            patchXb = fh.read('grid::xb')
        return patchXb
    
    def __toStartCount(self, coordinates, cellsPerPatch):
        """
        Transforms coordinate of lower left corner of patch and the number of grid cells contained in each patch to
        index of first particle in patch and the number of particles in the patch
        args:
            coordinates (int, int, int): lower left cell index of patch in grid
            cellsPerPatch [int]: number of cells per grid patch
        returns:
            offset of first particle in cell
            number of particles in cell
            idx of patch in grid meta-data
        """
        coord = np.array(coordinates)
        if( np.any(coord % cellsPerPatch != 0) ):
            raise Exception(f"Coordinates must be divisible by {cellsPerPatch}, patches contain {cellsPerPatch} cells in each dimension.")
            
        if self.patchCoordinates is not None and self.patchSizes is not None:
            idx = np.where( np.all(coord == self.patchCoordinates, axis=1) )[0][0]
            offsets = np.cumsum(self.patchSizes) - self.patchSizes
        else:
            raise Exception("Either patch coordinates or patch sizes not loaded")
        
        return [offsets[idx]], [self.patchSizes[idx]], idx
    
    def getPatchMomentum(self, coordinates, cellsPerPatch):
        """
        Return 3D momentum of particles in patch, normalized with mass = 1
        args:
            coordinates (int, int, int): lower left cell index of patch in grid
            cellsPerPatch [int]: number of cells per grid patch
        """
        #convert coordinates to patch index here
        
        start, count, _ = self.__toStartCount(coordinates, cellsPerPatch)

        with adios2.open(self.path, 'r') as fh:
            ux = fh.read('mprts::mprts::ux', start, count).reshape(-1,1)
            uy = fh.read('mprts::mprts::uy', start, count).reshape(-1,1)
            uz = fh.read('mprts::mprts::uz', start, count).reshape(-1,1)  
            kind = fh.read('mprts::mprts::kind', start, count).reshape(-1,1)
        
        prts = np.concatenate((ux, uy, uz, kind), axis=1)
        prts = pd.DataFrame(prts, columns=['ux', 'uy', 'uz', 'kind'], dtype=np.float32)
        if self.species != -1:
            prts = prts[prts['kind'] == self.species]
        
        return prts[['ux', 'uy', 'uz', 'kind']]
    
    def getPatchPosition(self, coordinates, cellsPerPatch):
        """
        Return 3D position of particles in patch
        args:
            coordinates (int, int, int): lower left cell index of patch in grid
            cellsPerPatch [int]: number of cells per grid patch

        """
        start, count, idx = self.__toStartCount(coordinates, cellsPerPatch)
        
        with adios2.open(self.path, 'r') as fh:
            x = fh.read('mprts::mprts::x', start, count).reshape(-1,1)
            y = fh.read('mprts::mprts::y', start, count).reshape(-1,1)
            z = fh.read('mprts::mprts::z', start, count).reshape(-1,1)
            kind = fh.read('mprts::mprts::kind', start, count).reshape(-1,1)
        prts = np.concatenate((x, y, z, kind), axis=1)
        prts = pd.DataFrame(prts, columns=['x', 'y', 'z', 'kind'], dtype=np.float32)
        prts[['x', 'y', 'z']] = prts[['x', 'y', 'z']] + self.xb[idx]
        
        if self.species != -1:
            prts = prts[prts['kind'] == self.species]
        
        return prts[['x', 'y', 'z', 'kind']]
    
    def getPatch(self, coordinates, cellsPerPatch):
        """
        Return all features of particles in patch
        args:
            coordinates (int, int, int): lower left cell index of patch in grid 
            cellsPerPatch [int]: number of cells per grid patch
        
        """ 
        start, count, idx = self.__toStartCount(coordinates, cellsPerPatch)
        
        with adios2.open(self.path, 'r') as fh:
            x = fh.read('mprts::mprts::x', start, count).reshape(-1,1)
            y = fh.read('mprts::mprts::y', start, count).reshape(-1,1)
            z = fh.read('mprts::mprts::z', start, count).reshape(-1,1)
            ux = fh.read('mprts::mprts::ux', start, count).reshape(-1,1)
            uy = fh.read('mprts::mprts::uy', start, count).reshape(-1,1)
            uz = fh.read('mprts::mprts::uz', start, count).reshape(-1,1)  
            kind = fh.read('mprts::mprts::kind', start, count).reshape(-1,1)
            qni_wni = fh.read('mprts::mprts::qni_wni', start, count).reshape(-1,1)
            
        prts = np.concatenate((x, y, z, ux, uy, uz, kind, qni_wni), axis=1)
        prts = pd.DataFrame(prts, columns=['x', 'y', 'z', 'ux', 'uy', 'uz', 'kind', 'qni_wni'], dtype=np.float32)
        prts[['x', 'y', 'z']] = prts[['x', 'y', 'z']] + self.xb[idx]
        
        if self.species != -1:
            prts = prts[prts['kind'] == self.species]
        
        return prts


class SuperCell(Particles):
    """
    Class to aggregate adjacent patches of particles (a Super Cell), used for analysis of particles in super cell.
    Uses JIT data read.  Selectively reads relevant particles belonging to super cell
    """
    def __init__(self, path, centroid, patches=3, cellsPerPatch=32, species='all'):
        """
        args:
            path: path [str]: path to checkpoint file
            centroid (int, int, int): lower left coordinates of patch which acts as the centroid of super cell.
            patches [int]: number of patches in each direction of centroid to build super cell.  Each dimension will be 2xpatches+1 wide
            cellsPerPatch [int]: number of grid cells in each patch
        """
        super().__init__(path, species)
        self.centroid = centroid
        self.patches = patches
        self.cellsPerPatch = cellsPerPatch
        self.residents = None
    def __collectPrts(self, getPositions=False):
        """
        Read particles belonging to super cell from checkpoint file. Only retrieve useful features
        args:
            centroid (int, int, int): lower left coordinates of centroid patch
            patches [int]: number of patches in each direction of centroid to build super cell.
            cellsPerPatch [int]: number of grid cells in each patch
            getPostitions [bool]: If true, read particle position data in addition to momentum 
        """
        patches = self.patches
        
        if patches % 2 == 1:
            lastPatch = 1
        else:
            lastPatch = 0
        patches = patches // 2
       
        x, y, z = self.centroid
        residents = []
        #for i in range(-patches, patches+1):  add for 3d supercell 
        for j in range(-patches, patches+lastPatch):
            for k in range(-patches, patches+lastPatch):
                
                patchX, patchY, patchZ = x, y + (j * self.cellsPerPatch), z + (k * self.cellsPerPatch)
                #patchX, patchY, patchZ = x + i*cellsPerPatch, y + j*cellsPerPatch, z + k*cellsPerPatch this line for 3d supercell
                
                if(
                    patchX >= 0 and patchX <= self.gridMaxDims[0] and
                    patchY >= 0 and patchY <= self.gridMaxDims[1] and
                    patchZ >= 0 and patchZ <= self.gridMaxDims[2]
                  ):
                    coordinates = (patchX, patchY, patchZ)
                    prts = self.getPatchMomentum(coordinates, self.cellsPerPatch)
                    
                    if getPositions:
                        positions = self.getPatchPosition(coordinates, self.cellsPerPatch)
                        prts = pd.concat((prts, positions), axis=1)
                        
                    residents.append( prts )
                    
        if len(residents) > 0:
            self.residents = pd.concat(residents, ignore_index=True)
        else:
            self.residents = pd.DataFrame(columns = self.columns)
        
        
    def histogram2D(self, bins):
        """
        Build and plot histogram of 2D particle log momentum, y and z directions. Normalized
        args:
            bins [int]: number of bins to be used in histogram construction
        """
        if self.residents is None:
            self.__collectPrts()
        
        #2d array y, z dims, might add switch to select which dimensions
        data = self.residents[['uy', 'uz']].to_numpy()

        hist = plt.hist2d(data[:,0],data[:,1], bins, norm=LogNorm())
        plt.colorbar(hist[3])
        
    def histogramV(self, bins, log=False, dim='s', savePath=None, slice_=None):
        """
        Construct and plot 1D histogram of particle V.  Normalized, optionally logarithmic.
        args:
            bins [int]: number of bins to be used in histogram construction
            log [bool]: Logarithmic bins
            dim [char]: 'x', 'y', 'z', 's', s=scalar product (vx^2 + vy^2 + vz^2)
            savePath[str/path]: path to save histogram data
            slice_[char]: 'x', 'y', 'z', dimension to filter data, only keep 3 sigma
        """
        if self.residents is None:
            self.__collectPrts(getPositions=bool(slice_))
            
        if bool(slice_):    
            self.residents = self.residents[ np.abs(stats.zscore(self.residents[slice_])) < 1]
        
        if dim == 's':
            data = 0.5*(self.residents['ux']**2 + self.residents['uy']**2 + self.residents['uz']**2)
        else:
            idx = {'x': 'ux', 'y': 'uy', 'z': 'uz'}[dim]
            data = self.residents[idx]
        ## better to return ndarray hist than to plot here 
        h = np.histogram(data, bins, density=False)
        if savePath:
            np.save(savePath, h)
        return h

    def momentT(self, bins: int, dim: str):
        """
        Calculate and plot particle Temperature moment along specified dimension.
        args:
            bins [int]: number of bins to be used in histogram construction
            dim [str]: string of dimension along which to calculate Temperature moment (x, y z)
        """
        if self.residents is None:
            self.__collectPrts(getPositions=True)

        idx = {'x': 'ux', 'y': 'uy', 'z': 'uz'}[dim]
        V = self.residents[idx]
        
        #build histogram based on L2 norm of velocity.
        n, bins, patches = self.histogramV(bins, log=False)
        if(np.sum(n*np.diff(bins)) != 1.0):
            raise Exception("Error: probability density function not achieved.")
            
        #get list of indices so that we know which bin the particle is in and thus the discrete
        #probability of finding particle with that velocity
        indices = np.digitize(V, bins)

        #calculate moments for each particle
        moments = V*V*n[indices]
        
        #plot
        X = self.residents['x'].to_numpy()
        Y = self.residents['y'].to_numpy()
        xmin, xmax, ymin, ymax = X.min(), X.max(), Y.min(), Y.max()
        xx = np.linspace(xmin, xmax, int(xmax-xmin)).reshape(1,-1)
        yy = np.linspace(ymin, ymax, int(ymax-ymin)).reshape(-1,1)
        zz = griddata((X, Y), moments, (xx, yy), method='nearest')
        plt.imshow(zz.T)
        plt.colorbar()
        






class Fields:
    def __init__(self, path):
        """
        args:
            path [str]: path to checkpoint file
        """
        self.path = path
        self.fieldMap = {'jx':0, 'jy':1, 'jz':2, 'ex':3, 'ey':4, 'ez':5, 'hx':6, 'hy':7, 'hz':8}
        
    def plot(self, field: str, corner: tuple, dims: tuple):
        """
        args:
            field [str]: field to plot, follows the following convention
                field map
                jx, jy, jz, ex, ey, ez, hx, hy, hz
                0    1   2   3   4   5   6   7   8
            corner [tuple/int]: tuple of ints, lower left hand corner of grid to load
            dims: [tuple/int]: dimensions of grid to load
        returns:
            None: plots field selection from checkpoint file
        """
        fdx = self.fieldMap[field]
        x, y, z = corner
        dimx, dimy, dimz= dims
        with adios2.open(self.path, 'r') as fh:
            checkpoint_fields = fh.read('mflds', [fdx, z, y, x], [1, dimz, dimy, dimx])
            #checkpoint_fields = fh.read('mflds', [fdx,1920,0,640], [1,400,640,1])
        
        plt.pcolormesh(checkpoint_fields[0,:,:,0],  shading='gouraud')
        plt.colorbar()   
        
        print(f'Grid Shape {checkpoint_fields.shape[1:]}')
