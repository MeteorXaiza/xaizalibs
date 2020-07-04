# coding:utf-8


import csv
import itertools

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


class PolyFittingManager():
    def __init__(self, deg=1):
        self.deg = deg
        self.arrCnt = None
        self.dicArrSumDivProdPowXAndPowYBySqErrY = None
        self.symbolN = sp.Symbol('n')
        self.dicSymbolSumDivProdPowXAndPowYBySqErrY = {}
        self.dicSymbolSumDivProdPowXAndPowYBySqErrY[(0, 0)] = sp.Symbol(
            'sum(1/sigma_i**2)')
        self.dicSymbolSumDivProdPowXAndPowYBySqErrY[(1, 0)] = sp.Symbol(
            'sum(x_i/sigma_i**2)')
        for cnt in range(2*self.deg - 1):
            degX = cnt + 2
            self.dicSymbolSumDivProdPowXAndPowYBySqErrY[(degX, 0)] = (
                sp.Symbol('sum(x_i**'+str(degX)+'/sigma_i**2)'))
        self.dicSymbolSumDivProdPowXAndPowYBySqErrY[(0, 1)] = sp.Symbol(
            'sum(y_i/sigma_i**2)')
        self.dicSymbolSumDivProdPowXAndPowYBySqErrY[(1, 1)] = sp.Symbol(
            'sum(x_i*y_i/sigma_i**2)')
        for cnt in range(self.deg - 1):
            degX = cnt + 2
            self.dicSymbolSumDivProdPowXAndPowYBySqErrY[(degX, 1)] = sp.Symbol(
                'sum(x_i**'+str(degX)+'*y_i/sigma_i**2)')
        self.dicSymbolSumDivProdPowXAndPowYBySqErrY[(0, 2)] = sp.Symbol(
            'sum(y_i**2/sigma_i**2)')
        self.dicSymbolParam = {}
        for cnt in range(self.deg + 1):
            self.dicSymbolParam[cnt] = sp.Symbol('a_'+str(cnt))
        self.symbolX = sp.Symbol('x_i')
        self.symbolY = sp.Symbol('y_i')
        self.symbolSigma = sp.Symbol('sigma_i')
        self.lsSymbolOptParam = []
        for cnt in range(self.deg + 1):
            self.lsSymbolOptParam.append(sp.Symbol('a_' + str(cnt)))
        self.tpShape = None
    def defineShape(self, tpShape):
        self.tpShape = tpShape
        self.arrCnt = np.zeros(tpShape, dtype='int64')
        self.dicArrSumDivProdPowXAndPowYBySqErrY = {}
        for cnt in range(2*self.deg + 1):
            self.dicArrSumDivProdPowXAndPowYBySqErrY[(cnt, 0)] = np.zeros(
                tpShape)
        for cnt in range(self.deg + 1):
            self.dicArrSumDivProdPowXAndPowYBySqErrY[(cnt, 1)] = np.zeros(
                tpShape)
        self.dicArrSumDivProdPowXAndPowYBySqErrY[(0, 2)] = np.zeros(tpShape)
    def appendData(self, arrX, arrY, arrErrY=None, arrIsValid=None):
        if self.tpShape is None:
            self.defineShape(arrX.shape)
        if arrErrY is None:
            arrErrY = np.ones(self.tpShape)
        if arrIsValid is None:
            arrIsValid = np.ones(self.tpShape, dtype='bool')
        self.arrCnt += arrIsValid.astype(int)
        self.dicArrSumDivProdPowXAndPowYBySqErrY[(0, 0)] += np.where(
            arrIsValid, 1/arrErrY**2, 0)
        for cnt in range(2*self.deg):
            degX = cnt + 1
            self.dicArrSumDivProdPowXAndPowYBySqErrY[(degX, 0)] += np.where(
                arrIsValid, arrX**degX/arrErrY**2, 0)
        self.dicArrSumDivProdPowXAndPowYBySqErrY[(0, 1)] += np.where(
            arrIsValid, arrY/arrErrY**2, 0)
        for cnt in range(self.deg):
            degX = cnt + 1
            self.dicArrSumDivProdPowXAndPowYBySqErrY[(degX, 1)] += np.where(
                arrIsValid, arrX**degX*arrY/arrErrY**2, 0)
        self.dicArrSumDivProdPowXAndPowYBySqErrY[(0, 2)] += np.where(
            arrIsValid, arrY**2/arrErrY**2, 0)
    def genSymbolFuncOptParam(self, deg):
        lsSymbolSumProdPowXAndWeight = []
        for cnt in range(2*self.deg + 1):
            lsSymbolSumProdPowXAndWeight.append(
                self.dicSymbolSumDivProdPowXAndPowYBySqErrY[(cnt, 0)])
        lsLsSymbolSumProdPowXAndWeight = []
        lsSymbolSumProdPowXAndYAndWeight = []
        for cnt in range(self.deg + 1):
            lsLsSymbolSumProdPowXAndWeight.append(
                lsSymbolSumProdPowXAndWeight[cnt:cnt+self.deg+1])
            lsSymbolSumProdPowXAndYAndWeight.append(
                self.dicSymbolSumDivProdPowXAndPowYBySqErrY[(cnt, 1)])
        matrixSymbolSumProdPowXAndWeight = sp.Matrix(
            lsLsSymbolSumProdPowXAndWeight)
        vectorSymbolSumProdPowXAndYAndWeight = sp.Matrix(
            lsSymbolSumProdPowXAndYAndWeight)
        symbolRet = (
            (matrixSymbolSumProdPowXAndWeight**-1)[deg, :]
            * vectorSymbolSumProdPowXAndYAndWeight
        )[0].factor()
        return symbolRet
    def genSymbolFuncSumFunc(self, symbolFunc):
        symbolFunc = symbolFunc.factor()
        symbolRet = 0
        for cnt in range(2*self.deg + 1):
            symbolRet += (
                symbolFunc.coeff(
                    self.symbolSigma, -2
                ).expand().coeff(
                    self.symbolX, cnt
                ).coeff(
                    self.symbolY, 0
                ) * self.dicSymbolSumDivProdPowXAndPowYBySqErrY[(cnt, 0)])
        for cnt in range(self.deg + 1):
            symbolRet += (
                symbolFunc.coeff(
                    self.symbolSigma, -2
                ).expand().coeff(
                    self.symbolX, cnt
                ).coeff(
                    self.symbolY, 1
                ) * self.dicSymbolSumDivProdPowXAndPowYBySqErrY[(cnt, 1)])
        symbolRet += (
            symbolFunc.coeff(
                self.symbolSigma, -2
            ).expand().coeff(
                self.symbolX, 0
            ).coeff(
                self.symbolY, 2
            ) * self.dicSymbolSumDivProdPowXAndPowYBySqErrY[(0, 2)])
        return symbolRet.factor()
    def genSymbolFuncErrFunc(self, symbolFunc):
        symbolFuncDiffFuncByY = 0
        for cnt in range(self.deg + 1):
            symbolFuncDiffFuncByY += (
                symbolFunc.diff(
                    self.dicSymbolSumDivProdPowXAndPowYBySqErrY[(cnt, 1)])
                * self.symbolX**cnt * self.symbolSigma**-2)
        symbolFuncDiffFuncByY += (
            symbolFunc.diff(
                self.dicSymbolSumDivProdPowXAndPowYBySqErrY[(0, 2)])
            * 2 * self.symbolY * self.symbolSigma**-2)
        symbolFuncProdSqDiffFuncByYAndSqErrY = (
            symbolFuncDiffFuncByY**2 * self.symbolSigma**2
        ).expand()
        return sp.sqrt(
            self.genSymbolFuncSumFunc(symbolFuncProdSqDiffFuncByYAndSqErrY))
    def genSymbolFuncErrParam(self, deg):
        return self.genSymbolFuncErrFunc(self.genSymbolFuncOptParam(deg))
    def genSymbolFuncSigmaY(self):
        symbolFuncRes = self.symbolY
        for cnt in range(self.deg + 1):
            symbolFuncRes -= self.lsSymbolOptParam[cnt] * self.symbolX**cnt
        symbolFuncDivSqResBySqErrY = (
            symbolFuncRes**2 / self.symbolSigma**2).expand()
        return sp.sqrt(
            (self.symbolN / (self.symbolN - (self.deg + 1)))
            * self.genSymbolFuncSumFunc(symbolFuncDivSqResBySqErrY)
            / self.dicSymbolSumDivProdPowXAndPowYBySqErrY[(0, 0)])
    def genArrFunc(self, symbolFunc):
        lsSymbol = []
        lsArr = []
        lsSymbol.append(self.symbolN)
        lsArr.append(self.arrCnt)
        for tpDeg in self.dicSymbolSumDivProdPowXAndPowYBySqErrY.keys():
            lsSymbol.append(self.dicSymbolSumDivProdPowXAndPowYBySqErrY[tpDeg])
            lsArr.append(self.dicArrSumDivProdPowXAndPowYBySqErrY[tpDeg])
        for cnt in range(self.deg + 1):
            lsSymbol.append(self.lsSymbolOptParam[cnt])
            if symbolFunc.diff(self.lsSymbolOptParam[cnt]) != 0:
                lsArr.append(self.genArrFunc(self.genSymbolFuncOptParam(cnt)))
            else:
                lsArr.append(None)
        lambdifyFunc = sp.lambdify(lsSymbol, symbolFunc)
        return lambdifyFunc(*lsArr)
    def genArrOptParam(self, deg):
        return self.genArrFunc(self.genSymbolFuncOptParam(deg))
    def genArrErrParam(self, deg, normalize=False):
        arrRet = self.genArrFunc(self.genSymbolFuncErrParam(deg))
        if normalize:
            arrRet *= self.genArrFunc(self.genSymbolFuncSigmaY())
        return arrRet
