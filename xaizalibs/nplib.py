# coding:utf-8


import numpy as np
import sympy as sp

from .standardlib import *


########################################
###########CSV関連#######################
########################################
def getArrFloatCsv(strFilePath, message=False):
    if message:
        print('loading ' + strFilePath + '...')
    File = open(strFilePath, "r")
    if message:
        print('finished.')
    ls = list(csv.reader(
        File, delimiter=",", doublequote=True, lineterminator="\r\n",
        quotechar='"', skipinitialspace=True))
    return np.array(ls).astype(float)


def getArrStrCsv(strFilePath, message=False):
    if message:
        print('loading ' + strFilePath + '...')
    File = open(strFilePath, "r")
    if message:
        print('finished.')
    ls = list(csv.reader(
        File, delimiter=",", doublequote=True, lineterminator="\r\n",
        quotechar='"', skipinitialspace=True))
    return np.array(ls)


def saveAsCsv(arr, strFilePath, encoding='utf-8', dir=True, message=False):
    #2次元ndarrayをcsv形式で保存する関数
    strFileAbsPath = getStrAbsPath(strFilePath)
    strDirAbsPath = genLsStrDirPathAndFileName(strFileAbsPath)[0]
    if not os.path.isdir(strDirAbsPath) and dir:
        mkdir(strDirPath, message=message)
    if np.ndim(arr) == 1:
        arr = np.reshape(arr, (1,np.size(arr)))
    with open(strFilePath, 'w', encoding=encoding) as File:
        writer = csv.writer(File,lineterminator="\n")
        for cnt in range(len(arr)):
            writer.writerow(arr[cnt])
    if message:
        print(strFileAbsPath + " has been saved.")


########################################
###########Numpy関連#####################
########################################
def genArrHueCircle(arrX):
    def red(index):
        if (0 <= index) and (index < 255):
            return 255
        elif (255 <= index) and (index < 510):
            return 510 - index
        elif (510 <= index) and (index < 1020):
            return 0
        elif (1020 <= index) and (index < 1275):
            return index - 1020
        elif (1275 <= index) and (index < 1530):
            return 255
    def green(index):
        if (0 <= index) and (index < 255):
            return index
        elif (255 <= index) and (index < 765):
            return 255
        elif (765 <= index) and (index < 1020):
            return 1020 - index
        elif (1020 <= index) and (index < 1530):
            return 0
    def blue(index):
        if (0 <= index) and (index < 510):
            return 0
        elif (510 <= index) and (index < 765):
            return index - 510
        elif (765 <= index) and (index < 1275):
            return 255
        elif (1275 <= index) and (index < 1530):
            return 1530 - index
    arrIndex = np.mod(arrX * (1530. / (2. * np.pi)), 1530).flatten()
    lsRed, lsGreen, lsBlue = [], [], []
    for cnt in range(arrIndex.size):
        index = arrIndex[cnt]
        lsRed.append(red(index))
        lsGreen.append(green(index))
        lsBlue.append(blue(index))
    ret = np.array([lsRed,lsGreen,lsBlue]).T
    lsShape = list(arrX.shape)
    lsShape.append(3)
    ret = ret.reshape(tuple(lsShape))
    return ret


def genArrNCR(n, r):
    return np.array(list(itertools.combinations(range(n), r)))


def genArrSumPowResFromDicArrSumPow(dicArrSumPowX, arrDec, deg, arrCntX):
    x, mu = sp.symbols('x mu')
    expandPolyFunc = sp.expand((x - mu) ** deg)
    fact = int(expandPolyFunc.coeff(x, 0).subs(mu, 1))
    arrRet = fact * arrDec**deg * arrCntX
    for cnt in range(deg):
        currentDeg = cnt + 1
        fact = int(expandPolyFunc.coeff(x, currentDeg).subs(mu, 1))
        arrInc = (
            fact * arrDec ** (deg - currentDeg)
            * dicArrSumPowX[currentDeg])
        if arrRet.dtype != arrInc.dtype:
            arrRet += arrInc.astype(arrRet.dtype)
        else:
            arrRet += arrInc
    return arrRet


def genArrMeanPowResFromDicArrSumPow(dicArrSumPowX, arrDec, deg, arrCntX):
    arrIsTargetPixelFrame = arrCntX > 0
    arrRet = (
        genArrSumPowResFromDicArrSumPow(dicArrSumPowX, arrDec, deg, arrCntX)
        / np.where(arrIsTargetPixelFrame, arrCntX, 1))
    arrRet[~arrIsTargetPixelFrame] = np.nan
    return arrRet


def genArrPowMomentFromDicArrSumPow(dicArrSumPowX, deg, arrCntX):
    # 仕様：dicArrSumPowXはkeyを次数とし、データの累乗の総和を内容とする辞書配列。
    # keyはint、内容はnumpy.ndarrayかつすべてのshapeが同じ。
    # keyは1からnまで存在している。
    # arrCntXはデータ数を内容とするnumpy.ndarrayでshapeが同じ
    arrIsValid = arrCntX > 0
    arrMeanX = dicArrSumPowX[1] / np.where(arrIsValid, arrCntX, 1)
    arrMeanX[~arrIsValid] = np.nan
    return genArrMeanPowResFromDicArrSumPow(
        dicArrSumPowX, arrMeanX, deg, arrCntX)


def genMeanPowResFromDicMeanPow(dicMeanPowX, dec, deg):
    ret = dec ** deg
    x, mu = sp.symbols('x mu')
    expandPolyFunc = sp.expand((x - mu) ** deg)
    for cnt in range(deg):
        currentDeg = cnt + 1
        fact = int(expandPolyFunc.coeff(x, currentDeg).subs(mu, 1))
        ret += (
            fact * dec ** (deg - currentDeg)
            * dicMeanPowX[currentDeg])
    return ret


def genArrStdFromDicArrSumPow(dicArrSumPowX, arrCntX, ddof=1):
    arrSqMoment = genArrPowMomentFromDicArrSumPow(dicArrSumPowX, 2, arrCntX)
    arrIsValid = ~np.isnan(arrSqMoment) * (arrCntX > ddof)
    arrRet = np.ones(dicArrSumPowX[1].shape) * np.nan
    arrValidCntX = arrCntX[arrIsValid]
    arrRet[arrIsValid] = np.sqrt(
        arrSqMoment[arrIsValid] * (arrValidCntX / (arrValidCntX - ddof)))
    return arrRet


def genArrSkewnessFromDicArrSumPow(dicArrSumPowX, arrCntX, ddof=1):
    arrCbMoment = genArrPowMomentFromDicArrSumPow(dicArrSumPowX, 3, arrCntX)
    arrStd = genArrStdFromDicArrSumPow(dicArrSumPowX, arrCntX, ddof=1)
    arrIsValid = ~np.isnan(arrCbMoment) * ~np.isnan(arrStd) * (arrStd != 0)
    arrRet = np.ones(dicArrSumPowX[1].shape) * np.nan
    arrRet[arrIsValid] = arrCbMoment[arrIsValid] / arrStd[arrIsValid]**3
    return arrRet


def genArrKurtosisFromDicArrSumPow(dicArrSumPowX, arrCntX, ddof=1):
    arrBiqMoment = genArrPowMomentFromDicArrSumPow(dicArrSumPowX, 4, arrCntX) #1000=>300
    arrStd = genArrStdFromDicArrSumPow(dicArrSumPowX, arrCntX, ddof=1) #600=>300
    arrIsValid = ~np.isnan(arrBiqMoment) * ~np.isnan(arrStd) * (arrStd != 0)
    arrRet = np.ones(dicArrSumPowX[1].shape) * np.nan
    arrRet[arrIsValid] = arrBiqMoment[arrIsValid]/arrStd[arrIsValid]**4 - 3
    return arrRet


def genArrBins(strBins, mean, std, skewness, kurtosis, min, max):
    return eval(strBins)
