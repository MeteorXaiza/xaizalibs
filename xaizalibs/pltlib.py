# coding:utf-8


import matplotlib.pyplot as plt
import numpy as np


########################################
###########matplot.pyplot関連############
########################################
def resetFig(height=640, width=480):
    plt.close()
    plt.figure(figsize=(height/100, width/100))


def setPlot(arrX, arrY, color='red', label=None, linewidth=None):
    plt.plot(arrX, arrY, color=color, label=label, linewidth=linewidth)


def setScatter(
        arrX, arrY, alpha=1., color='red', label=None, marker='.', s=36):
    plt.scatter(
        arrX, arrY, alpha=alpha, color=color,
        label=label, marker=marker, s=s)


def setErrorbar(arrX, arrY, arrYErr, color='red', fmt='none', label=None):
    plt.errorbar(arrX, arrY, arrYErr, color=color, fmt='none', label=label)


def setHist(
        arrData, bins=None, color='red', histtype='step', label=None, log=False,
        stacked=False):
    plt.hist(
        arrData, bins=bins, color=color, histtype=histtype, label=label,
        log=log, stacked=stacked)


def setHistFromVal(arrY, arrX, color='red', label=None, linewidth=1):
    arrHistYLow = np.zeros(arrX.size)
    arrHistYUp = np.zeros(arrX.size)
    arrHistYLow[1:] = arrY
    arrHistYUp[:-1] = arrY
    arrHistY = np.array([arrHistYLow, arrHistYUp]).T.flatten()
    arrHistX = np.array([arrX, arrX]).T.flatten()
    setPlot(arrHistX, arrHistY, color=color, label=label, linewidth=linewidth)


def setLegend(loc='best'):
    plt.legend(loc=loc)
