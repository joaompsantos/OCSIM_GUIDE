from matplotlib import pylab
import numpy as np
from numpy import fft
from bokeh.io import output_notebook, push_notebook
from bokeh.plotting import figure, show
from bokeh.layouts import row, column, gridplot
import bokeh.palettes as palettes
from scipy import signal
from scipy.io import wavfile
import warnings
from matplotlib import pylab as plt

# plot constellation
def plot_constellation(E,legend_label):
    if not isinstance(E,list):
        E = [E]
        
    if not isinstance(legend_label,list):
        legend_label = [legend_label]
    
    assert len(E) == len(legend_label), "Labels and signals should have the same length"
    
    fig = figure(output_backend="webgl",plot_height=300, plot_width=550)
    for s,l,c in zip(E,legend_label,palettes.turbo(len(E))):
        for i in np.arange(s.shape[0]):
            fig.scatter(s[i].real, s[i].imag, alpha=0.7,color=c, legend_label=l)
  
    fig.xaxis.axis_label = "In-Phase"
    fig.yaxis.axis_label = "Quadrature"
    #fig.legend.location = "top_left"
    fig.add_layout(fig.legend[0], 'right')
    show(fig)
    
# plot histogram
def plot_histogram(E,titlename,palette='Cividis256',mode = None):
    # palette='Viridis256'
    
    if mode == None:
        modes = np.arange(E.shape[0])
    else:
        if not isinstance(mode,list):
            modes = [mode]
        else:
            modes = mode
            
    for i in modes:
        
        plt.hist2d(np.reshape(E[i].real,(-1,)), np.reshape(E[i].imag,(-1,)), bins=200)
        plt.show()

        #H, xe, ye = np.histogram2d(np.reshape(E[i].real,(-1,)), np.reshape(E[i].imag,(-1,)), bins=80)
        #fig = figure(title='Mode '+str(i)+': '+titlename,plot_height=400, plot_width=400)
        #fig.image(image=[np.transpose(H)], x=xe[0], y=ye[0], dw=xe[-1] - xe[0], dh=ye[-1] - ye[0], palette=palette)

        #fig = figure(title=titlename, match_aspect=True,plot_height=400, plot_width=400)
        #fig.title.text_font_size = '16pt'
        #fig.grid.visible = False
        #r, bins = fig.hexbin(np.reshape(E.real,(-1,)), np.reshape(E.imag,(-1,)), size=0.015, hover_color="pink", hover_alpha=0.8, palette=palette)

        #show(fig)

# plot power Vs time
def plot_power_time(E,legend_label,L=100,mode = None):
    if not isinstance(E,list):
        E = [E]
        
    if not isinstance(legend_label,list):
        legend_label = [legend_label]
    
    assert len(E) == len(legend_label), "Labels and signals should have the same length"
    
    if mode == None:
        modes = np.arange(E[0].shape[0])
    else:
        if not isinstance(mode,list):
            modes = [mode]
        else:
            modes = mode
        
    for i in modes:
        fig1 = figure(title='Mode '+str(i)+': Real component',output_backend="webgl",plot_height=250, plot_width=900)

        for s,l,c in zip(E,legend_label,palettes.turbo(len(E))):
            Lnew = int(L*s.fs/s.fb)
            fig1.line(np.arange(Lnew)/s.fs,s.real[i,:Lnew],line_width=2, alpha=0.7,color=c, legend_label=l)

        fig1.xaxis.axis_label = "Time (s)"
        fig1.yaxis.axis_label = "Power"
        #fig1.legend.location = "top_left"
        fig1.add_layout(fig1.legend[0], 'right')

        fig2 = figure(title='Mode '+str(i)+': Imaginary component',output_backend="webgl",plot_height=250, plot_width=900)

        for s,l,c in zip(E,legend_label,palettes.turbo(len(E))):
            Lnew = int(L*s.fs/s.fb)
            fig2.line(np.arange(Lnew)/s.fs,s.imag[i,:Lnew],line_width=2, alpha=0.7,color=c, legend_label=l)

        fig2.xaxis.axis_label = "Time (s)"
        fig2.yaxis.axis_label = "Power"
        #fig2.legend.location = "top_left"
        fig2.add_layout(fig2.legend[0], 'right')

        show(column(fig1, fig2))

def plot_power_signal_time(E,legend_label,L=100,mode = None):
    if not isinstance(E,list):
        E = [E]
        
    if not isinstance(legend_label,list):
        legend_label = [legend_label]
    
    assert len(E) == len(legend_label), "Labels and signals should have the same length"
    
    if mode == None:
        modes = np.arange(E[0].shape[0])
    else:
        if not isinstance(mode,list):
            modes = [mode]
        else:
            modes = mode
        
    for i in modes:
        fig1 = figure(title='Mode '+str(i)+': Signal',output_backend="webgl",plot_height=250, plot_width=900)

        for s,l,c in zip(E,legend_label,palettes.turbo(len(E))):
            Lnew = int(L*s.fs/s.fb)
            
            norm = np.abs(s[i,:Lnew])
            phase = np.angle(s[i,:Lnew])
            t = np.arange(Lnew)/s.fs
            
            fig1.line(t,norm*np.cos(t*2*np.pi*s.fb + phase),line_width=2, alpha=0.7,color=c, legend_label=l)

        fig1.xaxis.axis_label = "Time (s)"
        fig1.yaxis.axis_label = "Power"
        #fig1.legend.location = "top_left"
        fig1.add_layout(fig1.legend[0], 'right')

        show(fig1)
        
# plot power Vs frequency
def plot_power_frequency(E,legend_label,mode = None):
    if not isinstance(E,list):
        E = [E]
        
    if not isinstance(legend_label,list):
        legend_label = [legend_label]
    
    assert len(E) == len(legend_label), "Labels and signals should have the same length"
    
    if mode == None:
        modes = np.arange(E[0].shape[0])
    else:
        if not isinstance(mode,list):
            modes = [mode]
        else:
            modes = mode
            
    for i in modes:
        fig3 = figure(title='Mode '+str(i),output_backend="webgl",plot_height=250, plot_width=900)

        for s,l,c in zip(E,legend_label,palettes.turbo(len(E))):    
            #mysig = np.atleast_2d(s)
            faxis = np.fft.fftfreq(s.shape[1],1/s.fs)
            psd = 20*np.log10(abs(np.fft.fft(s[i,:])))
            fig3.line(faxis,psd-psd.max(), line_width=2, alpha=0.7,color=c, legend_label=l)

        fig3.xaxis.axis_label = "Frequency"
        fig3.yaxis.axis_label = "Relative Power"
        #fig3.legend.location = "top_left"
        fig3.add_layout(fig3.legend[0], 'right')

        show(fig3)