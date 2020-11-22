from adios2toolspy import SuperCell, Particles, Fields
from h5toolspy import H5Processor
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plotGrid(root: str, fields: tuple, times: tuple, xrange: tuple=None, yrange: tuple=None, zrange: tuple=None):
    """
    plots list of fields at times in a nfields x ntimes grid
    args:
        root [path/str]: path to root directory
        fields [list/str]: list of strings representing fields to plot
        times [list/int]: list of times to plot, given as index
        (x/y/z)range [tuple/float]: coordinates to slice field
    """
    xmin, xmax = xrange if xrange else (None, None)
    ymin, ymax = yrange if yrange else (None, None)
    zmin, zmax = zrange if zrange else (None, None)

    nflds = len(fields)
    ntimes = len(times)
    
    h5p = H5Processor(root)
    
    fig, axes = plt.subplots(nrows=nflds, ncols=ntimes, squeeze=False)
    
    for fdx, fld in enumerate(fields):
        for tdx, time in enumerate(times):
            grid = h5p.getH5Grid(fld, time, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax)
            grid.plot(ax=axes[fdx,tdx])
            
def gridHist(root: str, field: str, time: int, dims: tuple,
             origin: tuple, species: str='electron', nbins: int=100, spacing:int=2,
             xrange: tuple=None, yrange: tuple=None, zrange: tuple=None):
    """
    overlay grid on field plot, construct associated histograms, plot in grid
    args:
        root [path/str]: path to root directory
        fields [list/str]: list of strings representing fields to plot
        times [list/int]: list of times to plot, given as index
        dims [list/int]: nrows x ncols to plot histogram grid
        (x/y/z)range [tuple/float]: coordinates to slice field
    """
    nrows, ncols = dims
    
    xmin, xmax = xrange if xrange else (None, None)
    ymin, ymax = yrange if yrange else (None, None)
    zmin, zmax = zrange if zrange else (None, None)
    
    fig, ax = plt.subplots(1)

    h5p = H5Processor(root)
    grid = h5p.getH5Grid(field, time, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax)
    grid.plot(ax=ax)
    
    corners, numPatches = h5p.getPatches(origin, nrows=nrows, ncols=ncols, spacing=spacing)
    
    c = np.arange(1,nrows*ncols + 1)
    norm = matplotlib.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet)
    cmap.set_array([])
    
    path = root + 'checkpoint_' + h5p.chkptTime + '.bp'
    
    fig2, axes2 = plt.subplots(nrows, ncols, sharex=True)
    for i in range(nrows):
        for j in range(ncols):
            SC = SuperCell(path, corners[i*ncols + j], patches=numPatches, species=species)
            h = SC.histogramV(nbins, dim='s')
            axes2[nrows-i-1, j].plot(h[1][1:], h[0], c=cmap.to_rgba(1+ i*ncols + j))
            axes2[nrows-i-1, j].set_yscale('log')
    plt.tight_layout()