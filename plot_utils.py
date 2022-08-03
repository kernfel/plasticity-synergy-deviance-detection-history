import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from brian2.units import msecond


# From http://www.ccctool.com/html_v_0_9_0_3/CCC_Tool/cccTool.html
ccc_divergent = LinearSegmentedColormap.from_list(
    'ccc_divergent', list(zip([0, .16, .35, .5, .62, .8, 1],
        np.asarray([
            [.0862745098039216,0.00392156862745098,0.298039215686275, 1.],
            [.054902,0.317647,0.709804, 1.],
            [.0705882,0.854902,0.870588, 1.],
            [1, 1, 1, 1.],
            [.94902,0.823529,0.321569, 1.],
            [.811765,0.345098,0.113725, 1.],
            [.188235294117647,0,0.0705882352941176, 1.]
        ]))))


def grouped_bars(series, xlabels, slabels, ax, w0=0.7):
    x = np.arange(len(xlabels))  # the label locations
    n = len(series)
    width = w0/n  # the width of the bars

    for i, (s, label) in enumerate(zip(series, slabels)):
        ax.bar(x - w0/2 + i*width, s, width, label=label)

    ax.set_xticks(x, xlabels)
    ax.legend()


def plot_pulse_hist(histograms, selection, tmax, dt, figsize=(10,15), grid=False, cmap='PiYG', vmin=None, vmax=None, symmetric=True, cscale=False):
    histograms = np.asarray(histograms)
    x = np.arange(tmax+1)*dt/msecond
    y = np.arange(len(selection)+1)
    if symmetric:
        if vmax is None:
            vmax = np.nanmax(np.abs(histograms))
        if vmin is None:
            vmin = -vmax
    else:
        if vmax is None:
            vmax = np.nanmax(histograms)
        if vmin is None:
            vmin = np.nanmin(histograms)

    fig, axs = plt.subplots(1, len(histograms), figsize=figsize, sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    axs = axs[0]
    orders = []
    for ax, hist in zip(axs, histograms):
        h = hist[selection, :tmax]
        order = 0
        if cscale:
            hmax, hmin = np.nanmax(h), np.nanmin(h)
            max_order = int(np.log10(vmax/hmax)) if hmax>0 else np.nan
            min_order = int(np.log10(vmin/hmin)) if hmin<0 else np.nan
            order = np.nanmin([max_order, min_order])
            if np.isnan(order):
                order = 1
        orders.append(order)
        m = ax.pcolormesh(x, y, h*10**order, vmin=vmin, vmax=vmax, cmap=cmap, shading='flat')
        ax.set_xlabel('Time after pulse onset (ms)')
        if grid:
            ax.grid()
    axs[0].set_ylabel('Neuron #')
    cb = plt.colorbar(m, location='bottom', ax=axs, aspect=40, fraction=1/figsize[1], pad=.5/figsize[1])

    if cscale:
        return fig, axs, cb, orders
    else:
        return fig, axs, cb