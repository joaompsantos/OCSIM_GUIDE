
import numpy as np
from numpy import array, abs, average, log10
import random
import time
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

###################
#				  #
# Receiver Blocks #
#				  #
###################

#
# Bit Sequence Generator
#
def bits( N, seed=None):
	if seed is None:
		random.seed(int(time.time()))  
	else:
		random.seed(seed)
			
	return array(random.choices( [0,1], k=N))


#
# Up-Sampling
#
def upsample( X, SpS):
	Y = np.zeros(SpS*len(X))
	Y[0::SpS] = X	
	return Y

#
# Pulse Shapping
#
def shapping( X, FILTER):
	h = filters(FILTER)
	X = np.convolve( X, h, 'full')
	return X, int(len(h/2))

#
# Laser 
#
def extmod( X, ER, Pavg):
	r = db2li(ER)
	P = db2li(Pavg)*1.1
	P0 = 2*P/(1+r)
	P1 = r*P0
	TX = np.interp( X, (X.min(), X.max()), (np.sqrt(P0), np.sqrt(P1)))
	return TX	

########################
#					   #	
# Linear Fiber Channel #
#					   #
########################

#
# EDFA
#
def edfa( X, G, Rb):
	h = 6.62607004e-34 
	G = db2li(G)-1
	nvar = h*193e12/2*2*Rb*G
	n = 4*np.sqrt(nvar)*np.random.randn(len(X))
	return X*np.sqrt(G)+n

#
# Fiber Attenuation
#
def fiber_attn( X, Ls, a=0.22):
	attn = db2li(a*Ls)
	return X/np.sqrt(attn)

###################
#				  #
# Receiver Blocks #
#				  #
###################

#
# Photodiode
#
def photodiode( X, R):
	q = 1.602176634e-19
	F = 10e9 
	k = 1.38e-23
	t = 298.15
	E = R*(abs(X)**2)*1e-3
	Qn = np.sqrt(2*q*F*E)*np.random.randn(len(X))
	Tn = np.sqrt(4*k*t*F/100)*np.random.randn(len(X))
	return E+Qn+Tn

#
# Avalanche Photodiode
#
def apd( X, R, M=12):
	q = 1.602176634e-19
	F = 10e9 
	k = 1.38e-23
	t = 298.15
	E = R*(abs(X)**2)*1e-3
	Qn = np.sqrt(2*q*F*E)*np.random.randn(len(X))*np.sqrt(M**2.1)
	return E*np.sqrt(M)+Qn

#
# Matched Filter
#
def rxfilter(X, FILTER):
	win = FILTER['SPS']*FILTER['LS']/2
	win = np.arange(0, win + 1)
	if FILTER['SHAPE'] != 'RC':
		win = filters(FILTER)
		X   = np.convolve(X, win,'full')

	return X, int(len(win))


# 
# Sampler
# 
def sampler( X, SpS, init=0):
	Y = X[init:len(X)-init]
	Y = Y[0::SpS]
	return Y

#
# Decision Block
#
def ber(X,Y):
	th = 0.5*(np.max(X)-np.min(X))+np.min(X)
	Y  = Y[0:len(X)]
	b  = scipy.optimize.fmin( minber, x0=[th], args=(X,Y), full_output=True, disp=False)
	return b

#
# Bit Error Rate
#
def minber(th,X,Y):
	B=X
	B[X>=th] = 1
	B[X<th]  = 0
	return (B != Y).mean()

###################### 
#					 #
# Auxiliar Functions #
#                    #
######################			

#
# Conversion of logarithmic to linear
#
def db2li(x):
	return 10 ** (x/10)

#
# Convertion of Linear to Logarithmic
#
def li2db(x):
	return 10*log10(x)

#
# Average Power
#
def avgpow(X):
	return average(abs(X**2))

#####################
#					#
# Figures and Plots #
#				    #
#####################

#
# Simple Eye Diagram
#
def eyediagram(X, SpS, delay):
	X = X[delay-1:-1-delay]
	neyes = int(np.floor(len(X)/SpS/2))
	eye = np.reshape(X[0:2*SpS*neyes],(neyes,-1))
	plt.grid()
	plt.plot(eye.T)
	plt.gca().set_aspect('auto')

def filters(FILTER):
	h=0
	if FILTER['SHAPE'] == 'RECT':
		h = np.ones(FILTER['SPS'])
	elif FILTER['SHAPE'] == 'SRRC':	
		SPS  = FILTER['SPS']
		BETA = FILTER['BETA']
		LS   = FILTER['LS']
		h = rcosdesign(BETA,LS,SPS,shape='sqrt')
	elif FILTER['SHAPE'] == 'RC':	
		SPS  = FILTER['SPS']
		BETA = FILTER['BETA']
		LS   = FILTER['LS']
		h = rcosdesign(BETA,LS,SPS,shape='normal')	
		
	return h

#
# RCOSDESIGN
#

def rcosdesign(beta, span, sps, shape='normal', dtype=np.float64):

    delay = span * sps / 2
    t = np.arange(-delay, delay + 1, dtype=dtype) / sps
    b = np.zeros_like(t)
    eps = np.finfo(dtype).eps

    if beta == 0:
        beta = np.finfo(dtype).tiny

    if shape == 'normal':
        denom = 1 - (2 * beta * t) ** 2

        ind1 = np.where(abs(denom) >  np.sqrt(eps), True, False)
        ind2 = ~ind1

        b[ind1] = np.sinc(t[ind1]) * (np.cos(np.pi * beta * t[ind1]) / denom[ind1]) / sps
        b[ind2] = beta * np.sin(np.pi / (2 * beta)) / (2 * sps)

    elif shape == 'sqrt':
        ind1 = np.where(t == 0, True, False)
        ind2 = np.where(abs(abs(4 * beta * t) - 1.0) < np.sqrt(eps), True, False)
        ind3 = ~(ind1 | ind2)

        b[ind1] = -1 / (np.pi * sps) * (np.pi * (beta - 1) - 4 * beta)
        b[ind2] = (
            1 / (2 * np.pi * sps)
            * (np.pi * (beta + 1) * np.sin(np.pi * (beta + 1) / (4 * beta))
            - 4 * beta * np.sin(np.pi * (beta - 1) / (4 * beta))
            + np.pi * (beta - 1) * np.cos(np.pi * (beta - 1) / (4 * beta)))
        )
        b[ind3] = (
            -4 * beta / sps * (np.cos((1 + beta) * np.pi * t[ind3]) +
                               np.sin((1 - beta) * np.pi * t[ind3]) / (4 * beta * t[ind3]))
            / (np.pi * ((4 * beta * t[ind3])**2 - 1))
        )

    else:
        raise ValueError('invalid shape')

    b /= np.sqrt(np.sum(b**2)) # normalize filter gain

    return b