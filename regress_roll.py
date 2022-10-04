#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:48:44 2020

@author: PatriciaCooney
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from loadconns_roll import loadconns_roll
#%%
### set up muscle activation targets ###

#generates a sine wave pulse from time pstart to time pend with a rise time of Trise
def genpulse(Tstop,dt,pstart,pend,Trise):
    #total time of pulse
    T = int(Tstop/dt)
    
    #initialize pulse that long
    p = np.zeros(T)
    
    istart = int(pstart/dt)
    iend = int(pend/dt)+1
    irise = int(Trise/dt)
    
    p[istart:iend] = 1.
    p[istart] = 0.
    p[istart:(istart+irise)] = np.sin(np.pi*np.arange(irise)/(2.*irise))
    p[(iend-irise):iend] = np.flipud(np.sin(np.pi*np.arange(irise)/(2.*irise)))
    
    return p

#%%
#generates targets for two segments given mnorder, the order of activations of different muscle groups
def gentargets(mnorder,dt):
    M = mnorder.shape[0]

    #define pulse parameters
    Tstop = 6
    Tpulse = 2
    dtpulse = 0.125 #time between start of one pulse and the next
    dtpulse_end = 0.125 #time between end of one pulse and the next
    tstart = 1
    T = int(Tstop/dt)

    pstarts = np.zeros([M])
    pends = np.zeros([M])

    for mi,ma in enumerate(mnorder):
        if ma == 0:
            pstarts[mi] = 0
            pends[mi] = 0
        else:
            pstarts[mi] = tstart + (mnorder[mi]-1)*dtpulse
            pends[mi] = tstart + Tpulse + (mnorder[mi]-1)*dtpulse_end

    mtarg = np.zeros([T,M])

    for mi in range(M):
        mtarg[:,mi] = genpulse(Tstop,dt,pstarts[mi],pends[mi],(pends[mi]-pstarts[mi])/2.)

    seg1inds = np.where(np.sum(mtarg[:,0:int(M/2)],1)>0)[0]
    seg2inds = np.where(np.sum(mtarg[:,int(M/2):M],1)>0)[0]

    return mtarg,seg1inds,seg2inds

#%%
#initializes connectivity matrix using signs of PMNs. For unknown neurotransmitter types, randomly assigns as excitatory or inhibitory.
def initJpp(Jpp0,types):
    J = np.zeros(Jpp0.shape,dtype=np.float32)
    N = J.shape[0]
    N2 = J.shape[1]
    J = np.copy(Jpp0)

    for qi in range(N):
        if types[qi] == 'inh':
            J[qi,:] = J[qi,:] * -1
        elif types[qi] == 'unknown': #randomly choose neurotransmitter sign if unknown
            J[qi,:] = J[qi,:] * np.random.choice([-1,1],N2)
    return J

#%%
#load data from hdf5 file, setup sine wave pulses, and initialize connectivity matrix

Jpm0,Jpp0,pnames,mnames,types,mnorder = loadconns_roll()
mnorder = mnorder.astype('float32')

M = len(mnames)
Mfit = np.nonzero(mnorder)[0]  #fit array of subset of M targets
N = len(pnames)
Ncycles = 1
dt = 0.05


mtarg,seg1inds,seg2inds = gentargets(mnorder,dt)




mtarg = mtarg[:,Mfit]
M = len(Mfit)

T = mtarg.shape[0]


Jpm0i = initJpp(Jpm0,types)

Jpm0i = Jpm0i[Mfit,:]

#%%
### do regression ###

from cvxopt import solvers, matrix

#minimize energy

#positive activations and MN threshold inequality
mz = np.array(np.where(mnorder == 0))
mz = mz[0,:]

G0 = -np.eye(N,N)
lambda_reg = 0.00001 #Ashok's code - 0.01

p = np.zeros([T,N])

gamma_smooth = 0.1

for ti in range(T):
    mtarg1 = mtarg[ti,:] #current target
    m_z = np.where(mtarg1 == 0)[0] #indices of MNs with zero target activity
    m_nz = np.setdiff1d(np.arange(M),m_z) #indices of MNs with nonzero target activity

    #set up equality cost
    if len(m_nz) > 0:
        Jpm0i_nz = np.copy(Jpm0i)
        Jpm0i_nz[m_z,:] = 0 #zero out rows corresponding to MNs with target activity = 0

        P = matrix(Jpm0i_nz.astype(float).T @  Jpm0i_nz.astype(float) + lambda_reg*np.eye(N))
        q = matrix(-2 * Jpm0i_nz.astype(float).T @ mtarg1)
    else: #no active MNs
        P = matrix(lambda_reg*np.eye(N))
        q = matrix(np.zeros([N,1]))

    #set up inequality constraints
    if len(m_z) > 0:
        G = matrix(np.vstack([G0,Jpm0i[m_z,:]])) #first M constraints enforce positive PMN firing rates, the remaining len(m_z constraints enforce nonpositive input to the MNs
        h = matrix(np.zeros([N+len(m_z),1]))
    else:
        G = matrix(G0)
        h = matrix(np.zeros([N,1]))

    p[ti,:] = np.array(solvers.qp(2*P, q, G, h)["x"]).flat
    #if ti > 0:
    #    for indx, xi in enumerate(p[ti,:]):
    #        p[ti,indx] = xi + gamma_smooth*((xi-p[ti-1,indx])**2)

paf = p.copy()

#%%
### make plots ###
Tx = len(seg1inds)
tx = np.arange(Tx)/Tx
Xmax = 1.

def findind(pstr):
    inds = [(pstr in x) for x in pnames]
    if np.sum(inds) == 0:
        print("error, no match for ",pstr)
        return -1
    elif np.sum(inds) > 1:
        print("error, multiple matches for ",pstr)
        return -1
    else:
        return np.where(inds)[0][0]


def findp(pstr):
    ind = findind(pstr)
    if ind == -1:
        return

    Jm = Jpm0[ind,:]
    Jpost = Jpp0[ind,:]
    Jpre = Jpp0[:,ind]

    minds = np.where(Jm > 0)[0]
    postinds = np.where(Jpost > 0)[0]
    preinds = np.where(Jpre > 0)[0]

    print(pnames[ind],types[ind])
    print("postsynaptic MNs:")
    for ii in minds:
        print("\t",mnames[ii],Jm[ii],"connections,",mnorder[:,ii])

    print("postsynaptic PMNs:")
    for ii in postinds:
        print("\t",pnames[ii],Jpost[ii],"connections,",types[ii])


    print("presynaptic PMNs:")
    for ii in preinds:
        print("\t",pnames[ii],Jpre[ii],"connections,",types[ii])

def plotact(pstr,finds,doav=True):
    ind = findind(pstr)

    plt.plot(tx,paf[finds,ind],color="k",lw=1)

def plotselected(pstrs,nrow,ncol,finds):
    Nplot = len(pstrs)
    for ii in range(Nplot):
        plt.subplot(nrow,ncol,ii+1)
        plotact(pstrs[ii],finds)
        plt.xticks((0,1))
        plt.xlim(0,Xmax)
        plt.title(pstrs[ii])

#%%
# for np,pn in enumerate(pnames):
#     plt.figure()
#     plt.plot(paf[:,np])
#     plt.title(pn)

plt.figure(figsize=(16,8))
pstrs = ["a27h_a1l","a27h_a1r","a18b3_a1l","a18b3_a1r","a31k_a1l","a31k_a1r","a01c1_a1l","a01c1_a1r","a14a_a1l","a14a_a1r","a18j_a1l","a18j_a1r","a23a_a1l","a23a_a1r","a18a_a1l","a18a_a1r","a02e_a1l","a02e_a1r"]
plotselected(pstrs,3,6,seg2inds)
plt.tight_layout()

plt.figure(figsize=(16,8))
pstrs = ["a27h_a2l","a27h_a2r","a18b3_a2l","a18b3_a2r","a31k_a2l","a31k_a2r","a01c1_a2l","a01c1_a2r","a14a_a2l","a14a_a2r","a18j_a2l","a18j_a2r","a23a_a2l","a23a_a2r","a18a_a2l","a18a_a2r","a02e_a2l","a02e_a2r"]
plotselected(pstrs,3,6,seg1inds)
plt.tight_layout()


#%%
x = range(0,120)

plt.figure
activity = p @ Jpm0i.T
activity = activity * (activity > 0)
plt.plot(x,activity,"r")

plt.figure
plt.plot(x,mtarg,"k")

# plt.savefig("attempt4.svg")

#%%
#Check how many MNs share common excitatory PMN input
# #find all excitatory PMNs in NT matrix and make new Jpm based on excitPMNs
# indexc = []

# for i,ii in enumerate(types):
#     if ii == 'exc':
#         indexc.append(i)
        
# excitJpm = Jpm0[:,indexc]

# ##make binary excitJpm matrix and store indices
# excitJpm[np.nonzero(excitJpm)] = 1

# ##make new matrix of MNxMN, where each element is shared input (# shared excit PMNs / # excit PMNs to self)
# #for MN(i) in matrix, for MN(ii) in matrix, find shared Jpm ones indices, calc shared input, store
# shared_input = np.zeros([104, 104]);
# perc_shared = np.zeros([104, 104]);

# i = 0


# while i < len(excitJpm)-1:
#     for j,jj in enumerate(excitJpm[i,:]):
#         if jj == 1:
#             k = 0
#             while k < len(excitJpm)-1:
#                 if excitJpm[k,j] == 1:
#                     shared_input[i,k] = shared_input[i,k] + 1
#                     perc_shared[i,k] = shared_input[i,k] / shared_input[i,i]
#                     k = k + 1
#                 else:
#                     k = k + 1
    
#     i = i + 1


# ##plot perc_shared matrix (rows and cols are MN #s and color of element is fraction of shared input (scaled 0 to 1))
# fig, ax = plt.subplots()

# min_val, max_val = 0, len(excitJpm)

# ax.matshow(perc_shared, cmap=plt.cm.Blues)

# ax.set_xlim(min_val, max_val)
# ax.set_ylim(min_val, max_val)
# ax.set_xticks(np.arange(max_val))
# ax.set_yticks(np.arange(max_val))
# plt.xticks(fontsize=2, rotation = 'vertical')
# plt.yticks(fontsize=2)
# plt.gca().invert_yaxis()
# ax.set_xticklabels(mnames)
# ax.set_yticklabels(mnames)

# plt.show()
# #fig.savefig('mnxmn-pmnexcitcnxns-new.svg', format='svg', dpi=1200)

# ##rank MNs by most to least shared inputs
# # df = pd.DataFrame(perc_shared, columns = mnames)

# # df = df.replace([np.inf, -np.inf], np.nan)
# # df = df.replace(1, np.nan)

# # desc_order = df.rank(ascending = 0)
# # desc_order = pd.DataFrame(desc_order, columns = mnames)
# # desc_order = df.rank(ascending = 0)


# ##print list of each MN's top 10 most shared connectivity
# topten = []

# for idx,mn in enumerate(mnames):
#     tempdf = pd.DataFrame(perc_shared[idx,:],mnames)
#     tempdf = np.transpose(tempdf)
#     tempdf = tempdf.replace([np.inf, -np.inf], np.nan)
#     tempdf = tempdf.replace(1, np.nan)
#     tempdf = np.array(tempdf)
#     # rankdf = tempdf.rank(axis = 1, method = 'max', ascending = False)
#     # rankdf = np.array(rankdf)
    
#     for i in range(0,10):
#         maxind = np.nanargmax(tempdf)
#         maxval = np.nanmax(tempdf)
#         maxname = mnames[maxind]
        
#         topten.append(maxname)
        
#         tempdf[0,maxind] = min_val #get rid of previous maximum value

# topten = np.reshape(topten,(104,10))
# topten = np.transpose(topten)

# toptendf = pd.DataFrame(topten, columns = mnames)

# #toptendf.to_csv('toptendf.csv', index = False)


# ##print list of each MN's fellow MNs that share 20% of excit PMN input
# nans = np.zeros([104,104]) + np.nan
# top = pd.DataFrame(nans, columns = None)

# for idx,mn in enumerate(mnames):
#     tempdf = pd.DataFrame(perc_shared[idx,:],mnames)
#     tempdf = np.transpose(tempdf)
#     tempdf = tempdf.replace([np.inf, -np.inf], np.nan)
#     tempdf = tempdf.replace(1, np.nan)
#     tempdf = np.array(tempdf)
#     # rankdf = tempdf.rank(axis = 1, method = 'max', ascending = False)
#     # rankdf = np.array(rankdf)
    
#     sharedpercmax = 1
#     sharedpercmin = 0.5 #change to whatever percent shared you desire
    
#     i = np.nanmax(tempdf)
#     count = 0
#     topperc = []
    
#     while i > sharedpercmin and i < sharedpercmax:
#         sharedpercind = np.nanargmax(tempdf)
#         sharedpercval = np.nanmax(tempdf)
#         sharedpercname = mnames[sharedpercind]
        
#         topperc.append(sharedpercname)
#         count = count + 1
        
#         tempdf[0,sharedpercind] = min_val #get rid of previous maximum value
#         i = np.nanmax(tempdf)
    
#     top.loc[0:count-1,idx] = topperc
#     top = top.rename(columns={idx : mnames[idx]})

# top.to_csv('top-50perc.csv', index = False)

