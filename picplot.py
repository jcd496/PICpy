from adios2toolspy import SuperCell
from h5toolspy import H5Processor
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fieldGrid(root: str, 
              fields: tuple,
              times: tuple,
              xrange: tuple=None,
              yrange: tuple=None,
              zrange: tuple=None,
              timeAveraged: bool=True,
              downsample: int = 0):
    """
    plots list of fields at times in a nfields x ntimes grid
    args:
        root [path/str]: path to root directory
        fields [list/str]: list of strings representing fields to plot
        times [list/int]: list of times to plot, given as index
        (x/y/z)range [tuple/float]: coordinates to slice field
        downsample [int]: sample every other point in each diminsion downsample number of times, ie dim/2^downsample
    """
    xmin, xmax = xrange if xrange else (None, None)
    ymin, ymax = yrange if yrange else (None, None)
    zmin, zmax = zrange if zrange else (None, None)

    nflds = len(fields)
    ntimes = len(times)
    
    h5p = H5Processor(root, timeAveraged, downsample)
    
    fig, axes = plt.subplots(nrows=nflds, ncols=ntimes, squeeze=False)
    
    for fdx, fld in enumerate(fields):
        for tdx, time in enumerate(times):
            grid = h5p.getH5Grid(fld, time, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax)
            grid.plot(ax=axes[fdx,tdx], cmap='RdBu_r')
            axes[fdx,tdx].set_title(fld)
    plt.tight_layout()
            
def histGrid(root: str, field: str, time: int, dims: tuple,
             origin: tuple, species: str='electron', nbins: int=100, spacing:int=2,
             cellsPerPatch: int=32, axes: tuple=('y', 'z'),
             xrange: tuple=None, yrange: tuple=None, zrange: tuple=None,
             timeAveraged: bool=True, downsample: int = 0):
    """
    overlay grid on field plot, construct associated histograms, plot in grid
    args:
        root [path/str]: path to root directory
        field [list/str]: field to overlay patch selections
        time [list/int]: time to plot, given as index
        dims [list/int]: nrows x ncols to plot histogram grid
        origin [list/float]: (xleft, ybottom) lower left corner coordinates of lower left most patch
        species [str]: electron, electron_he, or ion
        nbins [int]: number of bins to use in histogram construction
        spacing [int]: number of patches between histogram locations
        axes [tuple/str]: (xaxis dimension, yaxis dimension)
        cellsPerPatch [int]: number of grid cells in patch
        (x/y/z)range [tuple/float]: coordinates to slice field
        downsample [int]: sample every other point in each diminsion downsample number of times, ie dim/2^downsample
    """
    nrows, ncols = dims
    
    xmin, xmax = xrange if xrange else (None, None)
    ymin, ymax = yrange if yrange else (None, None)
    zmin, zmax = zrange if zrange else (None, None)

    fig, ax = plt.subplots(1)

    h5p = H5Processor(root, timeAveraged, downsample)
    grid = h5p.getH5Grid(field, time, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax)

    grid.plot(ax=ax)
    
    corners, numPatches = h5p.getPatches(origin, nrows=nrows, ncols=ncols, cellsPerPatch=cellsPerPatch, spacing=spacing, axes=axes)
    
    c = np.arange(1,nrows*ncols + 1)
    norm = matplotlib.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet)
    cmap.set_array([])

    path = root + 'checkpoint_' + h5p.chkptTime + '.bp'
    
    histograms = []
    fig2, axes2 = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False)
    for i in range(nrows):
        rowHistograms = []
        for j in range(ncols):
            SC = SuperCell(path, corners[i*ncols + j], cellsPerPatch=cellsPerPatch, patches=numPatches, species=species)
            h = SC.histogramV(nbins, dim='s')
            rowHistograms.append(h)
            axes2[nrows-i-1, j].scatter(h[1][1:], h[0], color=cmap.to_rgba(1+ i*ncols + j))
            axes2[nrows-i-1, j].set_yscale('log')
            
            ### ZERO OUT BINS WITH A SINGLE PARTICLE ###
            zero_out = h[0] <= 1
            h[0][zero_out] = 0
            
            popt, pcov = curve_fit(piecewise_linear, h[1][1:], np.log(h[0]+1))
            axes2[nrows-i-1, j].plot(h[1][1:], np.exp(piecewise_linear(h[1][1:], *popt)))
            axes2[nrows-i-1, j].set_title(f'Temps: {-1/popt[2]:.3f}, {-1/popt[3]:.3f}')
            
        histograms.insert(0, rowHistograms)
    fig2.tight_layout()
    
    return histograms


def piecewise_linear(x, x0, y0, k1, k2):
    y = np.piecewise(x, [x < x0, x >= x0],
                     [lambda x: k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0 ])
    return y