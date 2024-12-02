import os
import numpy as np
from numpy.fft import fftshift, ifftn, fftn

import matplotlib.pyplot as plt
csfont= {'fontname':'Times New Roman',
        'fontweight': 'bold'}
def k2x(kimg):
    return ifftn(fftshift(fftshift(kimg, 1,),0))

def x2k(img):
    return  fftshift(fftshift(fftn(img),1),0)

def syntheticGhost(img, ped, R=1):
    #ped: Change the PED direction
    # if LRRL:dir= 1, else APPA :dir=0
    if ped:
        img = img.T
        scalar_w = np.linspace(-1, 1, img.shape[0])[:, np.newaxis]
    else:
        scalar_w = np.linspace(-1,1, img.shape[1]).T[np.newaxis,:]
    kimg = x2k(img)
    kscale = kimg*scalar_w
    [kodd, keven] = [kscale.copy(),kscale.copy()]
    kodd[0*R::2,:] = 0
    keven[1*R::2,:] = 0
    imgSyn = k2x(kodd - keven)
    if ped:
        imgSyn = imgSyn.T
    return np.abs(imgSyn)

# --- Section below is for plotting images --- #

class myPlot():
    # default-setup
    def __init__(self,
                 saveDir = os.curdir,
                 figsize = (7,5),
                 origin_topleft = False,
                 blockShow = True,
                 overwriteALL = False):
        self.saveDir = saveDir
        self.figsize = figsize
        self.origin_topleft = origin_topleft
        self.blockshow = blockShow
        self.overwriteALL = overwriteALL
        self.hspace = 0
        self.wspace = 0.01
        self.titlepad = 0.8
        if self.origin_topleft:
            self.origin = 'lower'
        else:
            self.origin = 'upper'

    def saveFig(self, fig, figName='Figure', overwrite = False):
        if self.overwriteALL:
            overwrite=True
        figFN = os.path.join(self.saveDir, '{}.png'.format(figName))
        if os.path.exists(figFN):
            if not overwrite:
                print('Image Existed: {}'.format(figFN))
        print('Saving image to {}'.format(figFN))
        fig.savefig(figFN, bbox_inches = 'tight', transparent = True, pad_inches = 0)
        return figFN

    def show4imgs(self, listImgs,
                  listNames = None,
                  mainTitle = None,
                  figSize = None,
                  colorMap = 'viridis',
                  vmax = None,
                  vmin = None):
        if isinstance(listImgs, list):
            imgArr = np.squeeze(np.stack(listImgs, axis =-1))
        if isinstance(listImgs, np.ndarray):
            imgArr = listImgs
        if figSize is None: figSize = self.figsize
        n_imgs = imgArr.shape[-1]
        # show single img
        if len(imgArr.shape)==2:
            fig, axs = plt.subplots(1, 1, figsize = figSize)
            axs.imshow(imgArr, cmap = colorMap, vmax = vmax, vmin = vmin)
            axs.axis('off')
            if mainTitle: fig.suptitle(mainTitle)
            plt.show(block=self.blockshow)
            return fig
        n_rows = 1
        if n_imgs > 4: # default set to 4 images
            n_cols = 4
            n_rows = n_imgs//n_cols+1
        else:
            n_cols = n_imgs
        fig, axs = plt.subplots(n_rows, n_cols, figsize = figSize)
        if listNames is None:
            listNames = ['Fig {}'.format(i) for i in range(n_imgs)]
        axs = axs.flatten()
        for i in range(n_imgs):
            axs[i].imshow(imgArr[...,i], cmap = colorMap,
                          vmax = vmax, vmin = vmin)
            axs[i].axis('off')
            axs[i].set_title(listNames[i], **csfont)
        if mainTitle:
            fig.suptitle(mainTitle, y=self.titlepad, **csfont)
        plt.subplots_adjust(wspace=self.wspace, hspace=self.hspace)
        plt.show(block = self.blockshow)
        return fig




