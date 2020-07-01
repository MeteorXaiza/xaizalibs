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


class PolyFittingManager2():
    def __init__(self, deg=1):
        self.deg = deg
        self.dicArrSumDivProdPowXAndPowYBySqErrY = None
        self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY = {}
        self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(0, 0)] = sp.Symbol(
            'sum(1/sigma_i**2)')
        self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(1, 0)] = sp.Symbol(
            'sum(x_i/sigma_i**2)')
        for cnt in range(2*self.deg - 1):
            degX = cnt + 2
            self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(degX, 0)] = (
                sp.Symbol('sum(x_i**'+str(degX)+'/sigma_i**2)'))
        self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(0, 1)] = sp.Symbol(
            'sum(y_i/sigma_i**2)')
        self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(1, 1)] = sp.Symbol(
            'sum(x_i*y_i/sigma_i**2)')
        for cnt in range(self.deg - 1):
            degX = cnt + 2
            self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(degX, 1)] = sp.Symbol(
                'sum(x_i**'+str(degX)+'*y_i/sigma_i**2)')
        self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(0, 2)] = sp.Symbol(
            'sum(y_i**2/sigma_i**2')
        self.dicSymbolParam = {}
        for cnt in range(self.deg + 1):
            self.dicSymbolParam[cnt] = sp.Symbol('a_'+str(cnt))
        self.symbolX = sp.Symbol('x_i')
        self.symbolY = sp.Symbol('y_i')
        self.symbolSigma = sp.Symbol('sigma_i')
        self.tpShape = None
    def defineShape(self, tpShape):
        self.tpShape = tpShape
        self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY = {}
        for cnt in range(2*self.deg + 1):
            self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(cnt, 0)] = np.zeros(tpShape)
        for cnt in range(self.deg + 1):
            self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(cnt, 1)] = np.zeros(tpShape)
        self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(0, 2)] = np.zeros(tpShape)
    def appendData(self, arrX, arrY, arrErrY=None, arrIsValid=None):
        if self.tpShape is None:
            self.defineShape(arrX.shape)
        if arrErrY is None:
            arrErrY = np.ones(self.tpShape)
        if arrIsValid is None:
            arrIsValid = np.ones(self.tpShape, dtype='bool')
        self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(0, 0)] += np.where(
            arrIsValid, 1/arrErrY**2, 0)
        for cnt in range(2*self.deg):
            degX = cnt + 1
            self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(degX, 0)] += np.where(
                arrIsValid, arrX**degX/arrErrY**2, 0)
        self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(0, 1)] += np.where(
            arrIsValid, arrY/arrErrY**2, 0)
        for cnt in range(self.deg):
            degX = cnt + 1
            self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(degX, 1)] += np.where(
                arrIsValid, arrX**degX*arrY/arrErrY**2, 0)
        self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(0, 2)] = np.where(
            arrIsValid, arrY**2/arrErrY**2, 0)
    def genSymbolFuncOptParam(self, deg):
        lsSymbolSumProdPowXAndWeight = []
        for cnt in range(2*self.deg + 1):
            lsSymbolSumProdPowXAndWeight.append(
                self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(cnt, 0)])
        lsLsSymbolSumProdPowXAndWeight = []
        lsSymbolSumProdPowXAndYAndWeight = []
        for cnt in range(self.deg + 1):
            lsLsSymbolSumProdPowXAndWeight.append(
                lsSymbolSumProdPowXAndWeight[cnt:cnt+self.deg+1])
            lsSymbolSumProdPowXAndYAndWeight.append(
                self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(cnt, 1)])
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
        symbolRet = 0
        symbolRet += symbolFunc.coeff(self.symbolSigma, -2).subs([(self.symbolX, 0), (self.symbolY, 0)])
    def genSymbolFuncSumProdSqDiffFuncByYAndSqErrY(self, symbolFunc):
        symbolFuncDiffFuncByY = 0
        for cnt in range(self.deg + 1):
            symbolFuncDiffFuncByY += (
                symbolFunc.diff(
                    self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(cnt, 1)])
                * self.symbolX**cnt / self.symbolSigma**2)
        symbolFuncDiffFuncByY += (
            symbolFunc.diff(
                self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(0, 2)])
            * 2 * self.symbolY / self.symbolSigma**2)
        symbolFuncProdSqDiffFuncByYAndSqErrY = (
            symbolFuncDiffFuncByY * self.symbolSigma**2).expand()
        symbolRet = 0
        for cnt in range(2*self.deg + 1):
            symbolRet += (
                symbolFuncProdSqDiffFuncByYAndSqErrY.coeff(
                    self.symbolX, cnt
                ).coeff(
                    self.symbolY, 0
                ).coeff(
                    self.symbolSigma, -2
                ) * self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(cnt, 0)])
        for cnt in range(self.deg + 1):
            symbolRet += (
                symbolFuncProdSqDiffFuncByYAndSqErrY.coeff(
                    self.symbolX, cnt
                ).coeff(
                    self.symbolY, 1
                ).coeff(
                    self.symbolSigma, -2
                ) * self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(cnt, 1)])
        symbolRet += (
            symbolFuncProdSqDiffFuncByYAndSqErrY.coeff(
                self.symbolX, 0
            ).coeff(
                self.symbolY, 2
            ).coeff(
                self.symbolSigma, -2
            ) * self.dicSymbolSumDivProdPowXAndPowYAndBySqErrY[(0, 2)])








class PolyFittingManager():
    # x : 説明変数
    # y : 目的変数
    def __init__(self, deg=1):
        self.deg = deg
        self.dicArrSumPowX = None
        self.dicArrSumProdPowXAndY = None
        self.arrSumSqY = None
        self.tpShape = None
    def defineShape(self, tpShape):
        self.tpShape = tpShape
        self.dicArrSumPowX = {}
        self.dicArrSumProdPowXAndY = {}
        for cnt in range(2*self.deg + 1):
            self.dicArrSumPowX[cnt] = np.zeros(tpShape)
        for cnt in range(self.deg + 1):
            self.dicArrSumProdPowXAndY[cnt] = np.zeros(tpShape)
        self.arrSumSqY = np.zeros(tpShape)
    def appendData(self, arrX, arrY, arrIsValid=None):
        if self.tpShape is None:
            self.defineShape(arrX.shape)
        if arrIsValid is None:
            arrIsValid = np.ones(self.tpShape, dtype='bool')
        self.dicArrSumPowX[0] += np.where(arrIsValid, 1, 0)
        self.dicArrSumProdPowXAndY[0] += np.where(arrIsValid, arrY, 0)
        for cnt in range(2 * self.deg):
            self.dicArrSumPowX[cnt + 1] += np.where(arrIsValid, arrX**(cnt+1), 0)
        for cnt in range(self.deg):
            self.dicArrSumProdPowXAndY[cnt + 1] += np.where(
                arrIsValid, arrX**(cnt+1)*arrY, 0)
        self.arrSumSqY += np.where(arrIsValid, arrY**2, 0)
    def genSymbolFuncOptParam(self, deg):
        # self.deg=1, deg=1
        # => (sum(x**0)*sum(x**1*y) - sum(x**1)*sum(x**0*y))
        #    / (sum(x**0)*sum(x**2) - sum(x**1)**2)
        lsSymbolPowX = []
        for cnt in range(2*self.deg + 1):
            lsSymbolPowX.append(sp.Symbol('sum(x**'+str(cnt)+')'))
        lsMatrixSymbolPowX = []
        lsSymbolProdPowXAndY = []
        for cnt in range(self.deg + 1):
            lsMatrixSymbolPowX.append(lsSymbolPowX[cnt:cnt+self.deg+1])
            lsSymbolProdPowXAndY.append(sp.Symbol('sum(x**'+str(cnt)+'*y)'))
        matrixSymbolPowX = sp.Matrix(lsMatrixSymbolPowX)
        vectorSymbolProdPowXAndY = sp.Matrix(lsSymbolProdPowXAndY)
        symbolRet = (
            (matrixSymbolPowX**-1)[deg, :] * vectorSymbolProdPowXAndY
        )[0].factor()
        return symbolRet
    def genArrEvalSymbolFunc(self, symbolFunc):
        lsSymbol = []
        lsArrSum = []
        for cnt in range(2*self.deg + 1):
            lsSymbol.append(sp.Symbol('sum(x**'+str(cnt)+')'))
            lsArrSum.append(self.dicArrSumPowX[cnt])
        for cnt in range(self.deg + 1):
            lsSymbol.append(sp.Symbol('sum(x**'+str(cnt)+'*y)'))
            lsArrSum.append(self.dicArrSumProdPowXAndY[cnt])
        lsSymbol.append(sp.Symbol('sum(y**2)'))
        lsArrSum.append(self.arrSumSqY)
        lambdifyFunc = sp.lambdify(lsSymbol, symbolFunc.factor())
        return lambdifyFunc(*lsArrSum)
    def genArrOptParam(self, deg):
        return self.genArrEvalSymbolFunc(self.genSymbolFuncOptParam(deg))
    def genLsTpSymbolAndSymbolFuncOptParam(self):
        lsRet = []
        for cnt in range(self.deg + 1):
            lsRet.append(
                (sp.Symbol('a_'+str(cnt)), self.genSymbolFuncOptParam(cnt)))
        return lsRet
    def genSymbolFuncStdY(self, subs=False):
        # self.deg=1
        # => sqrt(
        #        (1 / (sum(x**0) - 2))
        #        * (
        #            a_0**2*sum(x**0) - 2*a_0*sum(x**0*y) + 2*a_0*a_1*sum(x**1)
        #            - 2*a_1*sum(x**1*y) + a_1**2*sum(x**2) + sum(y**2)))
        x = sp.Symbol('x')
        y = sp.Symbol('y')
        lsSymbolA = []
        for cnt in range(self.deg + 1):
            lsSymbolA.append(sp.Symbol('a_'+str(cnt)))
        symbolRes = y - lsSymbolA[0]
        for cnt in range(self.deg):
            symbolRes -= x**(cnt+1) * lsSymbolA[cnt+1]
        symbolSqRes = sp.expand(symbolRes ** 2)
        symbolSumSqRes = 0
        for cnt in range(2*self.deg + 1):
            symbolSumSqRes += (
                symbolSqRes.coeff(y, 0).coeff(x, cnt)
                * sp.Symbol('sum(x**'+str(cnt)+')'))
            symbolSumSqRes += (
                symbolSqRes.coeff(y, 1).coeff(x, cnt)
                * sp.Symbol('sum(x**'+str(cnt)+'*y)'))
        symbolSumSqRes += (
            symbolSqRes.coeff(y, 2).coeff(x, 0)
            * sp.Symbol('sum(y**2)'))
        symbolRet = sp.sqrt(symbolSumSqRes / (sp.Symbol('sum(x**0)') - 2))
        if subs:
            symbolRet = symbolRet.subs(
                self.genLsTpSymbolAndSymbolFuncOptParam())
        return symbolRet
    def genSymbolFuncSumSqDiffFuncByY(self, symbolFunc):
        symbolFuncDiffFuncByY = 0
        x, y = sp.symbols('x y')
        for cnt in range(self.deg + 1):
            symbolFuncDiffFuncByY += (
                symbolFunc.diff(sp.Symbol('sum(x**'+str(cnt)+'*y)')) * x**cnt)
        symbolFuncSqDiffFuncByY = (symbolFuncDiffFuncByY ** 2).expand()
        symbolRet = 0
        for cnt in range(2*self.deg + 1):
            symbolRet += (
                symbolFuncSqDiffFuncByY.coeff(x, cnt)
                * sp.Symbol('sum(x**'+str(cnt)+')'))
        return symbolRet.simplify()
    def genSymbolFuncErrParam(self, deg, subs=False):
        symbolFuncOptParam = self.genSymbolFuncOptParam(deg)
        symbolRet = (
            sp.Symbol('sigma_y')
            * sp.sqrt(self.genSymbolFuncSumSqDiffFuncByY(symbolFuncOptParam)))
        if subs:
            symbolRet = symbolRet.subs(
                sp.Symbol('sigma_y'), self.genSymbolFuncStdY(subs=True))
        return symbolRet
    def genArrErrParam(self, deg):
        lsTpSymbolAndSymbolFuncOptParam = []
        for cnt in range(self.deg + 1):
            lsTpSymbolAndSymbolFuncOptParam.append(
                (sp.Symbol('a_'+str(cnt)), self.genSymbolFuncOptParam(cnt)))
        symbolFuncErrParam = self.genSymbolFuncErrParam(deg, subs=True)
        return self.genArrEvalSymbolFunc(symbolFuncErrParam)
