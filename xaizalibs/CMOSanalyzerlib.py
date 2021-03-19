# coding:utf-8


from PIL import Image

import astropy.io.fits as ap
from astropy.visualization import ZScaleInterval

from .standardlib import *
from .nplib import *


########################################
###########FITSファイル関連###############
########################################
def genHDU(arr, header=None, primary=True):
    if primary:
        ret = ap.PrimaryHDU(arr)
    else:
        ret = ap.ImageHDU(arr)
    if header != None:
        if type(header) == dict:
            lsKey = header.keys()
        else:
            lsKey = range(len(header))
        for key in lsKey:
            ret.header[str(key)] = header[key]
    return ret


def getArrFits(strFilePath, header=False, index=0, message=False):
    if message:
        print('loading ' + strFilePath + '...')
    fits = ap.open(strFilePath)
    if message:
        print('finished.')
    if type(index) in [list,tuple]:
        lsIndex = index
        lsRet = [getArrFits(
            strFilePath, header=header, index=index, message=message
        ) for index in lsIndex]
        return type(lsIndex)(lsRet)
    else:
        hdu = fits[index]
        dicRet = {}
        dicRet['data'] = hdu.data
        dicRet['header'] = dict(hdu.header)
        if header:
            return dicRet
        else:
            return dicRet['data']


def getDicFits(strFilePath, index=0, message=False):
    dicRet = {}
    if message:
        print('loading ' + strFilePath + '...')
    fits = ap.open(strFilePath)
    if message:
        print('finished.')
    hdu = fits[index]
    dicRet['data'] = hdu.data
    dicRet['header'] = dict(hdu.header)
    return dicRet


def saveAsFits(arr, strFileName, header=None, message=False, divide=False):
    if divide:
        lsArr = list(arr)
        lsHDU = []
        cnt = 0
        primary = True
        for arr in lsArr:
            if header != None:
                lsHDU.append(genHDU(arr, header[cnt], primary=primary))
            else:
                lsHDU.append(genHDU(arr, primary=primary))
            cnt += 1
            primary = False
    else:
        lsHDU = [genHDU(arr, header)]
    hduList = ap.HDUList(lsHDU)
    hduList.writeto(strFileName, overwrite=True)
    if message:
        print(strFileName + " has been saved.")


def setHeader(arg):
    if arg == None:
        return 'None'
    else:
        return str(arg)


########################################
###########CMOS_Analyzer_version_5 関連##
########################################
def genArrArrPHArrange(maxLeak, mode='spiral'):
    arrRet = np.zeros(((2*maxLeak+1)**2, 2))
    arrRet[0] = maxLeak
    if mode == 'spiral':
        for cnt1 in range(maxLeak):
            for cnt2 in range(cnt1 + 2):
                arrRet[(2*cnt1+1)**2 + cnt2] = [maxLeak+cnt1+1, maxLeak+cnt2]
            for cnt2 in range(2*cnt1 + 1):
                arrRet[(2*cnt1 + 1)**2 + cnt1 + 2 + cnt2] = [
                    maxLeak+cnt1-cnt2, maxLeak+cnt1+1]
            for cnt2 in range(2*cnt1 + 3):
                arrRet[(2*cnt1 + 1)**2 + 3*cnt1 + 3 + cnt2] = [
                    maxLeak-cnt1-1, maxLeak+1+cnt1-cnt2]
            for cnt2 in range(2*cnt1 + 1):
                arrRet[(2*cnt1 + 1)**2 + 5*cnt1 + 6 + cnt2] = [
                    maxLeak-cnt1+cnt2, maxLeak-cnt1-1]
            for cnt2 in range(cnt1 + 1):
                arrRet[(2*cnt1 + 1)**2 + 7*cnt1 + 7 + cnt2] = [
                    maxLeak+cnt1+1, maxLeak-cnt1-1+cnt2]
    return arrRet.astype(int)


class BackGround():
    def __init__(
            self, tpFrameShape=None, lsDeg=[1,2,3,4],
            lsTpInvalidFrameShape=[], strDtype='float64'):
        self.tpFrameShape = None
        self.lsDeg = lsDeg
        self.lsTpInvalidFrameShape = lsTpInvalidFrameShape
        self.strDtype = strDtype
        self.lsFrameID = []
        self.lsStrSignalFrameFilePath = []
        self.arrCntSignalFrame = None
        self.dicArrSumPowSignalFrame = {}
        self.arrMinSignalFrame = None
        self.arrMaxSignalFrame = None
        if tpFrameShape is not None:
            self.defineFrameShape(tpFrameShape)
    def defineFrameShape(self, tpFrameShape):
        self.tpFrameShape = tpFrameShape
        self.arrCntSignalFrame = np.zeros(tpFrameShape).astype('int16')
        for deg in self.lsDeg:
            self.dicArrSumPowSignalFrame[deg] = (
                np.zeros(tpFrameShape).astype(self.strDtype))
        # np.nanはint型の配列にすると0になってしまうのでfloatで定義する
        self.arrMinSignalFrame = (
            np.ones(tpFrameShape).astype('float16') * np.nan)
        self.arrMaxSignalFrame = (
            np.ones(tpFrameShape).astype('float16') * np.nan)
    def invalidFrameShapeProcess(
            self, mode='continue', tpCurrentFrameShape=None, message=False,
            strSignalFrameFilePath=None):
        if message and strSignalFrameFilePath is not None:
            print(
                'WARNING : the shape of ' + strSignalFrameFilePath
                + ' is invalid.')
        if mode == 'continue':
            return False
        elif mode == 'quit':
            quit()
        elif mode == 'select':
            if tpCurrentFrameShape in self.lsTpInvalidFrameShape:
                return False
            strSelect = getStrSelect(strMessage='\n'.join([
                (
                    '1 : ' + str(self.tpFrameShape[0])
                    + 'x' + str(self.tpFrameShape[1])),
                (
                    '2 : ' + str(tpCurrentFrameShape[0])
                    + 'x' + str(tpCurrentFrameShape[1]))
            ]), lsStrValid=['1','2'])
            if strSelect == '1':
                return False
            else:
                return True
    def appendData(
            self, arrSignalFrame, frameID=None, eventData=None,
            strInvalidFrameShapeProcessMode='continue', maxLeak=None,
            message=False, ignoreNan=True, ignoreInf=True, ignoreNinf=True):
        if self.tpFrameShape is None:
            self.defineFrameShape(arrSignalFrame.shape)
        elif self.tpFrameShape != arrSignalFrame.shape:
            res = self.invalidFrameShapeProcess(
                mode=strInvalidFrameShapeProcessMode,
                tpCurrentFrameShape=arrSignalFrame.shape,
                message=message, strSignalFrameFilePath=strSignalFrameFilePath)
            if res:
                self.__init__(
                    tpFrameShape=arrSignalFrame.shape, lsDeg=self.lsDeg,
                    lsTpInvalidFrameShape=(
                        self.lsTpInvalidFrameShape+[self.tpFrameShape]))
            else:
                return None
        self.lsFrameID.append(frameID)
        # 無効な値（＝nan、inf、-inf）のマスク作成処理
        arrIsInvalidFrame = np.zeros(arrSignalFrame.shape, dtype='bool')
        if ignoreNan:
            arrIsInvalidFrame += np.isnan(arrSignalFrame)
        if ignoreInf:
            arrIsInvalidFrame += arrSignalFrame == np.inf
        if ignoreNinf:
            arrIsInvalidFrame += arrSignalFrame == -np.inf

        if eventData is not None:
            if maxLeak is None:
                maxLeak = eventData.maxLeak
            arrIsEventRangeFrame = eventData.genArrIsEventRangeFrame(
                tpFrameShape=self.tpFrameShape, maxLeak=maxLeak)
        if eventData is not None:
            self.arrCntSignalFrame[
                    ~arrIsEventRangeFrame * ~arrIsInvalidFrame
                ] += 1
        else:
            self.arrCntSignalFrame[~arrIsInvalidFrame] += 1
        # 累乗の処理
        if eventData is not None:
            arrReplaceSignalFrame = np.where(
                    arrIsEventRangeFrame+arrIsInvalidFrame, 0,
                    arrSignalFrame
                ).astype(self.strDtype)
            for deg in self.lsDeg:
                self.dicArrSumPowSignalFrame[deg] += arrReplaceSignalFrame ** deg
        else:
            arrTypedSignalFrame = np.where(
                    arrIsInvalidFrame, 0, arrSignalFrame
                ).astype(self.strDtype)
            for deg in self.lsDeg:
                self.dicArrSumPowSignalFrame[deg] += arrTypedSignalFrame ** deg
        # 最小値、最大値の処理
        # 初めて有効な信号が入ったピクセルの処理（＝最小値最大値をその信号に置換）
        if eventData is not None:
            arrIsTargetPixelFrame = (
                ~arrIsEventRangeFrame * ~arrIsInvalidFrame * (
                    self.arrCntSignalFrame == 1))
        else:
            arrIsTargetPixelFrame = ~arrIsInvalidFrame * (
                self.arrCntSignalFrame == 1)
        self.arrMinSignalFrame = np.where(
            arrIsTargetPixelFrame, arrSignalFrame, self.arrMinSignalFrame)
        self.arrMaxSignalFrame = np.where(
            arrIsTargetPixelFrame, arrSignalFrame, self.arrMaxSignalFrame)
        # 有効な信号が入ったピクセルかつこれが初めてではないピクセルの処理
        if eventData is not None:
            arrIsTargetPixelFrame = (
                ~arrIsEventRangeFrame * ~arrIsInvalidFrame * (
                    self.arrCntSignalFrame > 1))
        else:
            arrIsTargetPixelFrame = ~arrIsInvalidFrame * (
                self.arrCntSignalFrame > 1)
        arrTargetPixelSignal = arrSignalFrame[arrIsTargetPixelFrame]
        self.arrMinSignalFrame[arrIsTargetPixelFrame] = np.min([
            self.arrMinSignalFrame[arrIsTargetPixelFrame],
            arrTargetPixelSignal], axis=0)
        self.arrMaxSignalFrame[arrIsTargetPixelFrame] = np.max([
            self.arrMaxSignalFrame[arrIsTargetPixelFrame],
            arrTargetPixelSignal], axis=0)
    def loadSignalFrameFile(
            self, strSignalFrameFilePath, eventData=None, HDUIndex=0,
            strInvalidFrameShapeProcessMode='continue',
            maxLeak=None, message=False):
        arrSignalFrame = getArrFits(
            strSignalFrameFilePath, index=HDUIndex, message=message)
        if self.tpFrameShape is None:
            self.defineFrameShape(arrSignalFrame.shape)
        elif self.tpFrameShape != arrSignalFrame.shape:
            res = self.invalidFrameShapeProcess(
                mode=strInvalidFrameShapeProcessMode,
                tpCurrentFrameShape=arrSignalFrame.shape,
                message=message, strSignalFrameFilePath=strSignalFrameFilePath)
            if res:
                self.__init__(
                    tpFrameShape=arrSignalFrame.shape, lsDeg=self.lsDeg,
                    lsTpInvalidFrameShape=(
                        self.lsTpInvalidFrameShape+[self.tpFrameShape]))
            else:
                return None
        self.lsStrSignalFrameFilePath.append(strSignalFrameFilePath)
        if eventData is not None:
            if maxLeak is None:
                maxLeak = eventData.maxLeak
            arrIsEventRangeFrame = eventData.genArrIsEventRangeFrame(
                tpFrameShape=self.tpFrameShape, maxLeak=maxLeak)
        if eventData is not None:
            self.arrCntSignalFrame[~arrIsEventRangeFrame] += 1
        else:
            self.arrCntSignalFrame += 1
        # 累乗の処理
        if eventData is not None:
            arrReplaceSignalFrame = np.where(
                arrIsEventRangeFrame, 0, arrSignalFrame).astype(self.strDtype)
            for deg in self.lsDeg:
                self.dicArrSumPowSignalFrame[deg] += arrReplaceSignalFrame ** deg
        else:
            arrF64SignalFrame = arrSignalFrame.astype(self.strDtype)
            for deg in self.lsDeg:
                self.dicArrSumPowSignalFrame[deg] += arrF64SignalFrame ** deg
        # 最小値、最大値の処理
        if eventData is not None:
            # 初めて有効な信号が入ったピクセルの処理（＝最小値最大値をその信号に置換）
            arrIsTargetPixelFrame = ~arrIsEventRangeFrame * (
                self.arrCntSignalFrame == 1)
            self.arrMinSignalFrame = np.where(
                arrIsTargetPixelFrame, arrSignalFrame, self.arrMinSignalFrame)
            self.arrMaxSignalFrame = np.where(
                arrIsTargetPixelFrame, arrSignalFrame, self.arrMaxSignalFrame)
            # 有効な信号が入ったピクセルかつこれが初めてではないピクセルの処理
            arrIsTargetPixelFrame = ~arrIsEventRangeFrame * (
                self.arrCntSignalFrame > 1)
            arrTargetPixelSignal = arrSignalFrame[arrIsTargetPixelFrame]
            self.arrMinSignalFrame[arrIsTargetPixelFrame] = np.min([
                self.arrMinSignalFrame[arrIsTargetPixelFrame],
                arrTargetPixelSignal], axis=0)
            self.arrMaxSignalFrame[arrIsTargetPixelFrame] = np.max([
                self.arrMaxSignalFrame[arrIsTargetPixelFrame],
                arrTargetPixelSignal], axis=0)
        else:
            # 初めて有効な信号が入ったピクセルの処理（＝最小値最大値をその信号に置換）
            arrIsTargetPixelFrame = self.arrCntSignalFrame == 1
            self.arrMinSignalFrame = np.where(
                arrIsTargetPixelFrame, arrSignalFrame, self.arrMinSignalFrame)
            self.arrMaxSignalFrame = np.where(
                arrIsTargetPixelFrame, arrSignalFrame, self.arrMaxSignalFrame)
            # すべてのピクセルの最小値最大値を更新
            self.arrMinSignalFrame = np.min(
                [self.arrMinSignalFrame, arrSignalFrame], axis=0)
            self.arrMaxSignalFrame = np.max(
                [self.arrMaxSignalFrame, arrSignalFrame], axis=0)
    def genArrMeanPowSignalFrame(self, deg):
        arrIsTargetPixelFrame = self.arrCntSignalFrame > 0
        arrRet = (
            self.dicArrSumPowSignalFrame[deg]
            / np.where(arrIsTargetPixelFrame, self.arrCntSignalFrame, 1))
        arrRet[~arrIsTargetPixelFrame] = np.nan
        return arrRet
    def genArrStdSignalFrame(self, ddof=1):
        return genArrStdFromDicArrSumPow(
            self.dicArrSumPowSignalFrame,
            self.arrCntSignalFrame, ddof=ddof)
    def genArrSkewnessSignalFrame(self, ddof=1):
        return genArrSkewnessFromDicArrSumPow(
            self.dicArrSumPowSignalFrame,
            self.arrCntSignalFrame, ddof=ddof)
    def genArrKurtosisSignalFrame(self, ddof=1):
        return genArrKurtosisFromDicArrSumPow(
            self.dicArrSumPowSignalFrame,
            self.arrCntSignalFrame, ddof=ddof)
    def genArrSumPowPHFrame(self, arrZeroLevelFrame, deg):
        return genArrSumPowResFromDicArrSumPow(
            self.dicArrSumPowSignalFrame,
            arrZeroLevelFrame.astype(self.strDtype), deg,
            self.arrCntSignalFrame)
    def genArrMeanPowPHFrame(self, arrZeroLevelFrame, deg):
        return genArrPowMomentFromDicArrSumPow(
            self.dicArrSumPowSignalFrame, deg, self.arrCntSignalFrame)
    def genArrIsUnmaskedFrame(self, strValidPixelCondition):
        if 'Y' in strValidPixelCondition:
            Y = (
                np.arange(
                        self.tpFrameShape[0]
                    ).reshape((self.tpFrameShape[0], 1))
                * np.ones(self.tpFrameShape)).astype('int16')
        if 'X' in strValidPixelCondition:
            X = (
                np.arange(
                        self.tpFrameShape[1]
                    ).reshape((1, self.tpFrameShape[0]))
                * np.ones(self.tpFrameShape)).astype('int16')
        if 'mean' in strValidPixelCondition:
            mean = self.genArrMeanPowSignalFrame(1)
        if 'std' in strValidPixelCondition:
            std = self.genArrStdSignalFrame()
        if 'skewness' in strValidPixelCondition:
            skewness = self.genArrSkewnessSignalFrame()
        if 'kurtosis' in strValidPixelCondition:
            kurtosis = self.genArrKurtosisSignalFrame()
        if 'min' in strValidPixelCondition:
            min = self.arrMinSignalFrame
        if 'max' in strValidPixelCondition:
            max = self.arrMaxSignalFrame
        if 'cnt' in strValidPixelCondition:
            cnt = self.arrCntSignalFrame
        return eval(strValidPixelCondition)
    def genDicPHStats(self, arrZeroLevelFrame, arrIsUnmaskedFrame=None):
        dicMeanPowPH = {}
        if arrIsUnmaskedFrame is not None:
            cntSignal = self.arrCntSignalFrame[arrIsUnmaskedFrame].sum()
            arrIsTargetPixelFrame = (
                (self.arrCntSignalFrame > 0) * arrIsUnmaskedFrame)
            for deg in self.lsDeg:
                dicMeanPowPH[deg] = (
                    self.genArrSumPowPHFrame(arrZeroLevelFrame, deg)
                    / cntSignal)[arrIsUnmaskedFrame].sum()

        else:
            cntSignal = np.where(~np.isnan(arrZeroLevelFrame), self.arrCntSignalFrame, 0).sum()
            arrIsTargetPixelFrame = self.arrCntSignalFrame > 0
            for deg in self.lsDeg:
                dicMeanPowPH[deg] = np.where(
                    ~np.isnan(arrZeroLevelFrame),
                    (
                        self.genArrSumPowPHFrame(arrZeroLevelFrame, deg)
                        / cntSignal),
                    0
                ).sum()
        dicRet = {}
        if 1 in self.lsDeg:
            dicRet['mean'] = float(dicMeanPowPH[1])
            if 2 in self.lsDeg:
                dicRet['std'] = sqrt(
                    genMeanPowResFromDicMeanPow(
                        dicMeanPowPH, dicMeanPowPH[1], 2)
                    * (cntSignal - 1) / cntSignal)
                if 3 in self.lsDeg:
                    dicRet['skewness'] = float(
                        genMeanPowResFromDicMeanPow(
                            dicMeanPowPH, dicMeanPowPH[1], 3)
                        / dicRet['std']**3)
                    if 4 in self.lsDeg:
                        dicRet['kurtosis'] = float(
                            genMeanPowResFromDicMeanPow(
                                dicMeanPowPH, dicMeanPowPH[1], 4)
                            / dicRet['std']**4 - 3)
        dicRet['max'] = float((
            self.arrMaxSignalFrame - arrZeroLevelFrame
        )[arrIsTargetPixelFrame].max())
        dicRet['min'] = float((
            self.arrMinSignalFrame - arrZeroLevelFrame
        )[arrIsTargetPixelFrame].min())
        dicRet['cnt'] = int(cntSignal)
        return dicRet
    def genDicHeader(self, strFileType=''):
        dicRet = {}
        dicRet['FTYPE'] = strFileType
        for cnt in range(len(self.lsStrSignalFrameFilePath)):
            dicRet['F'+str(cnt)] = self.lsStrSignalFrameFilePath[cnt]
        return dicRet
    def saveMeanSignalFrameFile(
            self, strMeanSignalFrameFilePath, dicAppendixHeader={},
            message=False):
        dicHeader = self.genDicHeader('mean_BG')
        dicHeader.update(dicAppendixHeader)
        saveAsFits(
            self.genArrMeanPowSignalFrame(1), strMeanSignalFrameFilePath,
            header=dicHeader, message=message)
    def saveStdSignalFrameFile(
            self, strStdSignalFrameFilePath, dicAppendixHeader={},
            message=False):
        dicHeader = self.genDicHeader('std_BG')
        dicHeader.update(dicAppendixHeader)
        saveAsFits(
            self.genArrStdSignalFrame(), strStdSignalFrameFilePath,
            header=dicHeader, message=message)
    def saveSkewnessSignalFrameFile(
            self, strSkewnessSignalFrameFilePath, dicAppendixHeader={},
            message=False):
        dicHeader = self.genDicHeader('skewness_BG')
        dicHeader.update(dicAppendixHeader)
        saveAsFits(
            self.genArrSkewnessSignalFrame(), strSkewnessSignalFrameFilePath,
            header=dicHeader, message=message)
    def saveKurtosisSignalFrameFile(
            self, strKurtosisSignalFrameFilePath, dicAppendixHeader={},
            message=False):
        dicHeader = self.genDicHeader('kurtosis_BG')
        dicHeader.update(dicAppendixHeader)
        saveAsFits(
            self.genArrKurtosisSignalFrame(), strKurtosisSignalFrameFilePath,
            header=dicHeader, message=message)
    def saveMinSignalFrameFile(
            self, strMinSignalFrameFilePath, dicAppendixHeader={},
            message=False):
        dicHeader = self.genDicHeader('min_BG')
        dicHeader.update(dicAppendixHeader)
        saveAsFits(
            self.arrMinSignalFrame, strMinSignalFrameFilePath, header=dicHeader,
            message=message)
    def saveMaxSignalFrameFile(
            self, strMaxSignalFrameFilePath, dicAppendixHeader={},
            message=False):
        dicHeader = self.genDicHeader('max_BG')
        dicHeader.update(dicAppendixHeader)
        saveAsFits(
            self.arrMaxSignalFrame, strMaxSignalFrameFilePath, header=dicHeader,
            message=message)
    def saveCntSignalFrameFile(
            self, strCntSignalFrameFilePath, dicAppendixHeader={},
            message=False):
        dicHeader = self.genDicHeader('cnt_BG')
        dicHeader.update(dicAppendixHeader)
        saveAsFits(
            self.arrCntSignalFrame, strCntSignalFrameFilePath, header=dicHeader,
            message=message)


class EventData():
    def __init__(self):
        self.strSignalFrameFilePath = None
        self.maxLeak = None
        self.cnt = None
        self.arrArrCenterPixelIndex = None
        self.arrEvent_th = None
        self.arrSplit_th = None
        self.arrArrPHImg = None
        self.arrPHasum = None
        self.arrVortex = None
        self.arrArrArrAppendixImg = None
        self.successExtract = False
    def genArrIsEventRangeFrame(self, tpFrameShape, maxLeak=None):
        if maxLeak is None:
            maxLeak = self.maxLeak
        arrRet = np.zeros(tpFrameShape) != 0
        for y, x in self.arrArrCenterPixelIndex:
            arrRet[
                max(y-maxLeak, 0):min(y+maxLeak+1, tpFrameShape[0]),
                max(x-maxLeak, 0):min(x+maxLeak+1, tpFrameShape[1])] = True
        return arrRet
    def invalidFrameShapeProcess(
            mode='continue', message=False, strSignalFrameFilePath=None):
        if message and strSignalFrameFilePath is not None:
            print(
                'WARNING : the shape of ' + strSignalFrameFilePath
                + ' is invalid.')
        if mode == 'continue':
            return False
        elif mode == 'quit':
            quit()
    def extractFromSignalFrameFile(
            self, strSignalFrameFilePath, event_th, split_th,
            arrZeroLevelFrame=None, HDUIndex=0, lsArrAppendixFrame=[],
            maxLeak=1, message=False, strInvalidFrameShapeProcessMode='continue'
            ):
        self.strSignalFrameFilePath = strSignalFrameFilePath
        self.maxLeak = maxLeak
        arrSignalFrame = getArrFits(
            strSignalFrameFilePath, index=HDUIndex, message=message)
        # arrZeroLevelFrameの設定
        if arrZeroLevelFrame is None:
            arrZeroLevelFrame = np.zeros(arrSignalFrame.shape)
        if arrSignalFrame.shape != arrZeroLevelFrame.shape:
            res = self.invalidFrameShapeProcess(
                mode=strInvalidFrameShapeProcessMode,
                message=message, strSignalFrameFilePath=strSignalFrameFilePath)
            if not(res):
                return None
        # event_thの設定
        if type(event_th) == np.ndarray:
            arrEvent_thFrame = event_th
        else:
            arrEvent_thFrame = np.ones(arrSignalFrame.shape) * event_th
        # split_thの設定
        if type(split_th) == np.ndarray:
            arrSplit_thFrame = split_th
        else:
            arrSplit_thFrame = np.ones(arrSignalFrame.shape) * split_th
        arrPHFrame = arrSignalFrame - arrZeroLevelFrame
        arrIsEventCenterPixelFrame = np.ones(arrSignalFrame.shape, dtype=bool)
        arrIsEventCenterPixelFrame[:maxLeak, :] = False
        arrIsEventCenterPixelFrame[-maxLeak:, :] = False
        arrIsEventCenterPixelFrame[:, :maxLeak] = False
        arrIsEventCenterPixelFrame[:, -maxLeak:] = False
        arrIsEventCenterPixelFrame *= arrPHFrame > arrEvent_thFrame
        for cnt1 in range(maxLeak*2 + 1):
            for cnt2 in range(maxLeak*2 + 1):
                if cnt1 == maxLeak and cnt2 == maxLeak:
                    continue
                arrStopPixel = (
                    (
                        np.array(arrPHFrame.shape)
                        - 2*maxLeak
                        + np.array([cnt1, cnt2])
                    ).astype(int))
                arrIsEventCenterPixelFrame[
                    maxLeak:-maxLeak, maxLeak:-maxLeak
                ] *= (
                    arrPHFrame[maxLeak:-maxLeak, maxLeak:-maxLeak]
                    > arrPHFrame[cnt1:arrStopPixel[0], cnt2:arrStopPixel[1]])
        self.arrArrCenterPixelIndex = np.argwhere(arrIsEventCenterPixelFrame)
        self.cnt = self.arrArrCenterPixelIndex.shape[0]
        if message:
            prints(
                'event_count : ' + str(self.cnt))
        self.arrArrPHImg = (
            np.ones((self.cnt, 2*self.maxLeak+1, 2*self.maxLeak+1)) * np.nan)
        self.arrPHasum = np.ones(self.cnt) * np.nan
        self.arrVortex = np.ones(self.cnt) * np.nan
        self.arrArrArrAppendixImg = (
            np.ones((
                    len(lsArrAppendixFrame),
                    self.cnt, 2*self.maxLeak+1,
                    2*self.maxLeak+1))
            * np.nan)
        arrScoreTrimedImg = 2**(np.array([
            [6, 5 ,4],
            [7, 1, 3],
            [8, 1, 2]]).astype('int16') - 1)
        arrScoreTrimedImg[1, 1] = 0
        startPixel = maxLeak - 1
        stopPixel = maxLeak + 2
        self.arrEvent_th = np.zeros(self.cnt, dtype='float64')
        self.arrSplit_th = np.zeros(self.cnt, dtype='float64')
        for cnt1, arrEventCentePixel in enumerate(self.arrArrCenterPixelIndex):
            self.arrEvent_th[cnt1] = arrEvent_thFrame[
                arrEventCentePixel[0], arrEventCentePixel[1]]
            self.arrSplit_th[cnt1] = arrSplit_thFrame[
                arrEventCentePixel[0], arrEventCentePixel[1]]
            arrEventStartPixel = arrEventCentePixel - maxLeak
            arrEventStopPixel = arrEventCentePixel + maxLeak + 1
            self.arrArrPHImg[cnt1] = arrPHFrame[
                arrEventStartPixel[0] : arrEventStopPixel[0],
                arrEventStartPixel[1] : arrEventStopPixel[1]]
            arrSplit_thImg = arrSplit_thFrame[
                arrEventStartPixel[0] : arrEventStopPixel[0],
                arrEventStartPixel[1] : arrEventStopPixel[1]]
            arrIsExceeded = self.arrArrPHImg[cnt1] > arrSplit_thImg
            self.arrPHasum[cnt1] = np.where(
                arrIsExceeded, self.arrArrPHImg[cnt1], 0).sum()
            self.arrVortex[cnt1] = (
                np.where(
                    arrIsExceeded[
                        startPixel:stopPixel,startPixel:stopPixel],
                    arrScoreTrimedImg, 0).sum())
            for cnt2 in range(len(lsArrAppendixFrame)):
                self.arrArrArrAppendixImg[cnt2, cnt1] = (lsArrAppendixFrame[cnt2][
                    arrEventStartPixel[0] : arrEventStopPixel[0],
                    arrEventStartPixel[1] : arrEventStopPixel[1]])
        self.successExtract = True


class EventList():
    def __init__(self):
        self.lsEventData = []
    def appendEventData(
            self, eventData, strInvalidFrameShapeProcessMode='continue'):
        if eventData.successExtract:
            self.lsEventData.append(eventData)
        else:
            if strInvalidFrameShapeProcessMode == 'continue':
                return None
            elif strInvalidFrameShapeProcessMode == 'quit':
                quit()
    def getFrameNum(self, strSignalFrameFilePath, moveDir=True):
        for cnt, eventData in enumerate(self.lsEventData):
            if strSignalFrameFilePath == eventData.strSignalFrameFilePath:
                return cnt
        if not(moveDir):
            return None
        strTargetSignalFrameFileName = (
            genLsStrDirPathAndFileName(strSignalFrameFilePath)[1])
        for cnt, eventData in enumerate(self.lsEventData):
            strSourceSignalFrameFileName = genLsStrDirPathAndFileName(
                eventData.strSignalFrameFilePath)[1]
            if strSourceSignalFrameFileName == strTargetSignalFrameFileName:
                return cnt
        return None
    def genLsFrameNum(self):
        lsRet = []
        for cnt, eventData in enumerate(self.lsEventData):
            lsRet += [cnt] * eventData.cnt
        return lsRet
    def genLsStrSignalFrameFilePath(self):
        lsRet = []
        for eventData in self.lsEventData:
            lsRet.append(eventData.strSignalFrameFilePath)
        return lsRet
    def genLsMaxLeak(self):
        lsRet = []
        for eventData in self.lsEventData:
            lsRet.append(eventData.maxLeak)
        return lsRet
    def genLsCnt(self):
        lsRet = []
        for eventData in self.lsEventData:
            lsRet.append(eventData.cnt)
        return lsRet
    def genArrArrCenterPixelIndex(self):
        lsRet = []
        for eventData in self.lsEventData:
            lsRet += list(eventData.arrArrCenterPixelIndex)
        return np.array(lsRet)
    def genLsArrPHImg(self):
        lsRet = []
        for eventData in self.lsEventData:
            lsRet += eventData.arrArrPHImg
        return lsRet
    def genLsPHasum(self):
        lsRet = []
        for eventData in self.lsEventData:
            lsRet += list(eventData.arrPHasum)
        return lsRet
    def genLsVortex(self):
        lsRet = []
        for eventData in self.lsEventData:
            lsRet += list(eventData.arrVortex)
        return lsRet
    def genLsLsArrAppendixImg(self):
        lsRet = []
        for eventData in self.lsEventData:
            lsRet += eventData.lsLsArrAppendixImg
        return lsRet
    def genArrEventList(self):
        maxRequiredWidth =  15
        requiredHeight = 0
        for eventData in self.lsEventData:
            requiredWidth = (
                1 + 2 + 1 + 1 + 1
                + (
                    (2*eventData.maxLeak+1)**2
                    * (eventData.arrArrArrAppendixImg.shape[0] + 1)))
            # frame_num, pixel_index, max_leak, PHasum, vortex, PH_img,
            # appendix_img
            maxRequiredWidth = max(maxRequiredWidth, requiredWidth)
            requiredHeight += eventData.cnt
        arrRet = np.ones((requiredHeight, maxRequiredWidth)) * np.nan
        if requiredHeight <= 0:
            return arrRet
        arrRet[:, 0] = self.genLsFrameNum()
        arrRet[:, [1,2]] = self.genArrArrCenterPixelIndex()
        arrRet[:, 4] = np.array(self.genLsPHasum())
        arrRet[:, 5] = np.array(self.genLsVortex())
        startY = 0
        for eventData in self.lsEventData:
            stopY = startY + eventData.cnt
            arrRet[startY:stopY, 3] = eventData.maxLeak
            arrArrPHArrange = genArrArrPHArrange(eventData.maxLeak)
            arrArrPHImg = eventData.arrArrPHImg
            for cnt, arrPHArrange in enumerate(arrArrPHArrange):
                arrRet[startY:stopY, 6+cnt] = (
                    arrArrPHImg[:, arrPHArrange[0], arrPHArrange[1]])
            for cnt1, arrArrAppendixImg in enumerate(
                    eventData.arrArrArrAppendixImg):
                for cnt2, arrPHArrange in enumerate(arrArrPHArrange):
                    arrRet[startY:stopY, 6+(2*eventData.maxLeak+1)**2*(cnt1+1)+cnt2] = (
                        arrArrAppendixImg[:, arrPHArrange[0], arrPHArrange[1]])
            startY = stopY
        return arrRet
    def saveAsEventListFile(
            self, strEventListFilePath, dicAppendixHeader={}, message=False):
        arrEventList = self.genArrEventList()
        if message and arrEventList.shape[0] == 0:
            print('WARNING : event count is 0.')
        dicHeader = {}
        for cnt, eventData in enumerate(self.lsEventData):
            dicHeader['F' + str(cnt)] = eventData.strSignalFrameFilePath
        for key in dicAppendixHeader.keys():
            dicHeader[str(key)] = dicAppendixHeader[key]
        saveAsFits(
            arrEventList, strEventListFilePath, header=dicHeader,
            message=message)
    def genLsVal(self, strFilledVal='PHasum'):
        lsRet = []
        for eventData in self.lsEventData:
            max_leak = eventData.maxLeak
            arrArrPHImg = eventData.arrArrPHImg
            arrArrArrAppendixImg = eventData.arrArrArrAppendixImg
            arrArrPHArrange = genArrArrPHArrange(eventData.maxLeak)
            arrPH = (
                np.ones((eventData.cnt, (2*eventData.maxLeak+1)**2)) * np.nan)
            arrArrAppendix = (
                np.ones(
                    (
                        len(eventData.arrArrArrAppendixImg),
                        eventData.cnt,
                        (2*eventData.maxLeak+1)**2)
                ) * np.nan)
            for cnt1, arrPHArrange in enumerate(arrArrPHArrange):
                arrPH[:, cnt1] = (
                    arrArrPHImg[:, arrPHArrange[0], arrPHArrange[1]])
                for cnt2 in range(len(eventData.arrArrArrAppendixImg)):
                    arrArrAppendix[cnt2, :, cnt1] = arrArrArrAppendixImg[
                        cnt2, :, arrPHArrange[0], arrPHArrange[1]]
            for cnt in range(eventData.cnt):
                Y, X = eventData.arrArrCenterPixelIndex[cnt]
                PH = arrPH[cnt]
                PHasum = eventData.arrPHasum[cnt]
                vortex = eventData.arrVortex[cnt]
                appendix = arrArrAppendix[:, cnt]
                filledVal = eval(strFilledVal)
                lsRet.append(filledVal)
        return lsRet
    def loadEventListFile(self, strEventListFilePath, message=False):
        dicEventList = getDicFits(strEventListFilePath, message=message)
        arrEventList = dicEventList['data']
        dicHeader = dicEventList['header']
        startY = 0
        for cnt1 in range(int(arrEventList[-1,0]) + 1):
            eventData = EventData()
            eventData.strSignalFrameFilePath = dicHeader['F'+str(cnt1)]
            eventData.maxLeak = int(arrEventList[startY, 3])
            eventData.cnt = int((arrEventList[:, 0] == cnt1).sum())
            eventData.arrPHasum = arrEventList[startY:startY+eventData.cnt, 4]
            eventData.arrVortex = (
                arrEventList[startY:startY+eventData.cnt, 5].astype(int))
            eventData.arrArrCenterPixelIndex = arrEventList[startY:startY+eventData.cnt, 1:3].astype(int)
            arrArrPHImg = (
                np.ones((
                        eventData.cnt,
                        2*eventData.maxLeak+1,
                        2*eventData.maxLeak+1))
                * np.nan)
            arrArrPHArrange = genArrArrPHArrange(eventData.maxLeak)
            for cnt2, arrPHArrange in enumerate(arrArrPHArrange):
                arrArrPHImg[:, arrPHArrange[0], arrPHArrange[1]] = arrEventList[
                    startY:startY+eventData.cnt, 6+cnt2]
            eventData.arrArrPHImg = arrArrPHImg
            appendixCnt = (
                int((arrEventList.shape[1] - 6) / (2*eventData.maxLeak+1)**2)
                - 1)
            eventData.arrArrArrAppendixImg = (
                np.ones((
                        appendixCnt, eventData.cnt, eventData.maxLeak*2+1,
                        eventData.maxLeak*2+1))
                * np.nan)
            startX = 6 + (2*eventData.maxLeak + 1)**2
            for cnt2 in range(appendixCnt):
                arrArrAppendixImg = (
                    np.ones((
                            eventData.cnt,
                            2*eventData.maxLeak+1,
                            2*eventData.maxLeak+1))
                    * np.nan)
                for cnt3, arrPHArrange in enumerate(arrArrPHArrange):
                    arrArrAppendixImg[:, arrPHArrange[0], arrPHArrange[1]] = (
                        arrEventList[startY:startY+eventData.cnt, startX+cnt3])
                eventData.arrArrArrAppendixImg[cnt2] = arrArrAppendixImg
                startX += (2*eventData.maxLeak + 1) ** 2
            eventData.successExtract = True
            self.appendEventData(eventData)
            startY += eventData.cnt
    def genArrSpectrumVal(self, strFilledVal, strBins, strValidEventCondition):
        arrFilledVal = np.array(self.genLsVal(strFilledVal))
        if strValidEventCondition is not None:
            arrIsValid = np.array(self.genLsVal(strValidEventCondition))
        else:
            arrIsValid = np.zeros(arrFilledVal.size) == 0
        arrValidFilledVal = arrFilledVal[arrIsValid]
        mean = arrValidFilledVal.mean()
        std = arrValidFilledVal.std(ddof=1)
        skewness = ((arrValidFilledVal - mean) ** 3).mean() / std**3
        kurtosis = ((arrValidFilledVal - mean) ** 4).mean() / std**4 - 3
        min = arrValidFilledVal.min()
        max = arrValidFilledVal.max()
        arrBins = eval(strBins)
        lsHistVal = np.histogram(arrValidFilledVal, arrBins)
        arrRet = np.zeros((2,arrBins.size))
        arrRet[0][:-1] = lsHistVal[0]
        arrRet[1] = lsHistVal[1]
        return arrRet


class FrameStats():
    def __init__(self, lsDeg=[1,2,3,4], lsTpInvalidFrameShape=[]):
        self.tpFrameShape = None
        self.lsTpInvalidFrameShape = lsTpInvalidFrameShape
        self.frameCnt = 0
        self.lsDeg = lsDeg
        self.cnt = 0
        self.dicSumPow = {}
        for deg in lsDeg:
            self.dicSumPow[deg] = 0.
        self.max = None
        self.min = None
        self.tpFrameShape = None
        self.lsStrRawFrameFilePath = []
        self.arrCntFrame = None
        self.dicArrSumPowFrame = {}
        self.arrMinFrame = None
        self.arrMaxFrame = None
    def invalidFrameShapeProcess(
            self, mode='continue', message=False, strRawFrameFilePath=None):
        if message and strRawFrameFilePath is not None:
            print(
                'WARNING : the shape of ' + strRawFrameFilePath
                + ' is invalid.')
        if mode == 'continue':
            return False
        elif mode == 'quit':
            quit()
    def defineFrameShape(self, tpFrameShape):
        self.tpFrameShape = tpFrameShape
        self.arrCntSignalFrame = np.zeros(tpFrameShape).astype('int16')
        for deg in self.lsDeg:
            self.dicArrSumPowFrame[deg] = np.zeros(tpFrameShape)
        # np.nanはint型の配列にすると0になってしまうのでfloatで定義する
        self.arrMinFrame = (
            np.ones(tpFrameShape).astype('float16') * np.nan)
        self.arrMaxFrame = (
            np.ones(tpFrameShape).astype('float16') * np.nan)
    def loadFrameFile(
            self, strRawFrameFilePath, strFilledVal=None,
            strValidPixelCondition=None, message=False, ignoreNan=True,
            ignoreInf=True, ignoreNinf=True, eventData=None,
            strInvalidFrameShapeProcessMode='continue', tpValidFrameShape=None,
            lsArrReferenceFrame=[], excludeRim=True):
        arrRawFrame = getArrFits(strRawFrameFilePath, message=message)
        if self.tpFrameShape is None:
            self.defineFrameShape(arrRawFrame.shape)
        self.tpFrameShape = arrRawFrame.shape
        if tpValidFrameShape is not None:
            if self.tpFrameShape != tpValidFrameShape:
                res = self.invalidFrameShapeProcess(
                    strInvalidFrameShapeProcessMode, message=message,
                    strRawFrameFilePath=strRawFrameFilePath)
                if not(res):
                    return None
        if strFilledVal is not None:
            arrFilledValFrame = self.genArrValFrame(
                strFilledVal, arrRawFrame, ref=lsArrReferenceFrame)
        else:
            arrFilledValFrame = arrRawFrame
        if arrFilledValFrame is None:
            res = self.invalidFrameShapeProcess(
                strInvalidFrameShapeProcessMode, message=message,
                strRawFrameFilePath=strRawFrameFilePath)
            if not(res):
                return None
        if strValidPixelCondition is not None:
            arrIsValidPixelFrame = self.genArrValFrame(
                strValidPixelCondition, arrRawFrame, ref=lsArrReferenceFrame)
        else:
            arrIsValidPixelFrame = np.ones(self.tpFrameShape, dtype=bool)
        if arrIsValidPixelFrame is None:
            res = self.invalidFrameShapeProcess(
                strInvalidFrameShapeProcessMode, message=message,
                strRawFrameFilePath=strRawFrameFilePath)
            if not(res):
                return None
        if arrIsValidPixelFrame.shape != arrIsValidPixelFrame.shape:
            res = self.invalidFrameShapeProcess(
                strInvalidFrameShapeProcessMode, message=message,
                strRawFrameFilePath=strRawFrameFilePath)
            if not(res):
                return None
        if eventData is not None:
            arrIsValidPixelFrame *= ~eventData.genArrIsEventRangeFrame(
                arrRawFrame.shape)
            if excludeRim:
                arrIsValidPixelFrame[:2*eventData.maxLeak - 1, :] = False
                arrIsValidPixelFrame[-2*eventData.maxLeak + 1:, :] = False
                arrIsValidPixelFrame[:, :2*eventData.maxLeak - 1] = False
                arrIsValidPixelFrame[:, -2*eventData.maxLeak + 1:] = False
        if ignoreNan:
            arrIsValidPixelFrame *= ~np.isnan(arrFilledValFrame)
        if ignoreInf:
            arrIsValidPixelFrame *= arrFilledValFrame != np.inf
        if ignoreNinf:
            arrIsValidPixelFrame *= arrFilledValFrame != -np.inf
        arrValidFilledVal = (
            arrFilledValFrame[arrIsValidPixelFrame].astype('float64'))
        self.frameCnt += 1
        self.lsStrRawFrameFilePath.append(strRawFrameFilePath)
        self.cnt += arrValidFilledVal.size
        for deg in self.lsDeg:
            self.dicSumPow[deg] += (arrValidFilledVal ** deg).sum()
        if self.min is None:
            self.min = arrValidFilledVal.min()
        if self.max is None:
            self.max = arrValidFilledVal.max()
        self.min = min(self.min, arrValidFilledVal.min())
        self.max = max(self.max, arrValidFilledVal.max())
        for deg in self.lsDeg:
            self.dicArrSumPowFrame[deg] += np.where(arrIsValidPixelFrame, arrFilledValFrame, 0).astype('float64')
    def genArrValFrame(self, strVal, raw, ref=[]):
        Y = (
            np.arange(self.tpFrameShape[0]).reshape((self.tpFrameShape[0], 1))
            * np.ones(self.tpFrameShape))
        X = (
            np.arange(self.tpFrameShape[1]).reshape((1, self.tpFrameShape[1]))
            * np.ones(self.tpFrameShape))
        arrRet = eval(strVal)
        return arrRet
    def genDicMeanPow(self):
        dicRet = {}
        for deg in self.lsDeg:
            if self.cnt == 0:
                dicRet[deg] = np.nan
            else:
                dicRet[deg] = self.dicSumPow[deg] / self.cnt
        return dicRet
    def genStd(self, ddof=1):
        if not(1 in self.lsDeg and 2 in self.lsDeg):
            return None
        if self.cnt < ddof:
            return np.nan
        dicMeanPow = self.genDicMeanPow()
        return sqrt(
            genMeanPowResFromDicMeanPow(dicMeanPow, dicMeanPow[1], 2)
            * self.cnt / (self.cnt - 1))
    def genSkewness(self, ddof=1):
        if not(1 in self.lsDeg and 2 in self.lsDeg and 3 in self.lsDeg):
            return None
        if self.cnt < ddof:
            return np.nan
        std = self.genStd(ddof=ddof)
        if std == 0:
            return np.nan
        dicMeanPow = self.genDicMeanPow()
        return (
            genMeanPowResFromDicMeanPow(dicMeanPow, dicMeanPow[1], 3) / std**3)
    def genKurtosis(self, ddof=1):
        if not(
                1 in self.lsDeg and 2 in self.lsDeg
                and 3 in self.lsDeg and 4 in self.lsDeg):
            return None
        if self.cnt < ddof:
            return np.nan
        std = self.genStd(ddof=ddof)
        if std == 0:
            return np.nan
        dicMeanPow = self.genDicMeanPow()
        return (
            genMeanPowResFromDicMeanPow(dicMeanPow, dicMeanPow[1], 4)/std**4
            - 3)
    def genDicFrameStats(self):
        dicRet = {}
        dicMeanPow = self.genDicMeanPow()
        dicRet['sum_pow'] = {}
        dicRet['mean_pow'] = {}
        for deg in self.lsDeg:
            dicRet['sum_pow'][str(deg)] = self.dicSumPow[deg]
            dicRet['mean_pow'][str(deg)] = dicMeanPow[deg]
        dicRet['min'] = self.min
        dicRet['max'] = self.max
        dicRet['std'] = self.genStd()
        dicRet['skewness'] = self.genSkewness()
        dicRet['kurtosis'] = self.genKurtosis()
        dicRet['cnt'] = self.cnt
        dicRet['frame'] = self.lsStrRawFrameFilePath
        dicRet['frame_cnt'] = self.frameCnt
        return dicRet
    def saveAsFrameStatsFile(
            self, strFrameStatsFilePath, message=False, dicAppendix={}):
        dicFrameStats = self.genDicFrameStats()
        dicFrameStats.update(dicAppendix)
        saveAsJSON(
            dicFrameStats, strFrameStatsFilePath, message=message, indent=2)
    def loadFrameStatsFile(self, strFrameStatsFilePath, message=False):
        dicFrameStats = getDicJSON(strFrameStatsFilePath, message=message)
        self.cnt = dicFrameStats['cnt']
        self.lsDeg = []
        self.dicSumPow = {}
        for strDeg in dicFrameStats['sum_pow'].keys():
            deg = float(strDeg)
            self.lsDeg.append(deg)
            self.dicSumPow[deg] = dicFrameStats['sum_pow'][strDeg]
        self.min = dicFrameStats['min']
        self.max = dicFrameStats['max']


class FrameSpectrum():
    def __init__(self):
        self.arrBins = None
        self.arrCnt = None
        self.lsStrRawFrameFilePath = []
        self.frameCnt = 0
    def defineBins(self, strBins, frameStats):
        mean = frameStats.dicSumPow[1] / frameStats.cnt
        std = frameStats.genStd()
        skewness = frameStats.genSkewness()
        kurtosis = frameStats.genKurtosis()
        min = frameStats.min
        max = frameStats.max
        self.arrBins = eval(strBins)
        self.arrCnt = np.zeros(self.arrBins.size - 1)
    def invalidFrameShapeProcess(
            self, mode='continue', message=False, strRawFrameFilePath=None):
        if message and strRawFrameFilePath is not None:
            print(
                'WARNING : the shape of ' + strRawFrameFilePath
                + ' is invalid.')
        if mode == 'continue':
            return False
        elif mode == 'quit':
            quit()
    def loadFrameFile(
            self, strRawFrameFilePath, strFilledVal=None,
            strValidPixelCondition=None, message=False, ignoreNan=True,
            ignoreInf=True, ignoreNinf=True,
            strInvalidFrameShapeProcessMode='continue', lsArrReferenceFrame=[],
            tpValidFrameShape=None, eventData=None, excludeRim=True):
        arrRawFrame = getArrFits(strRawFrameFilePath, message=message)
        self.tpFrameShape = arrRawFrame.shape
        if tpValidFrameShape is not None:
            if self.tpFrameShape != tpValidFrameShape:
                res = self.invalidFrameShapeProcess(
                    strInvalidFrameShapeProcessMode, message=message,
                    strRawFrameFilePath=strRawFrameFilePath)
                if not(res):
                    return None
        if strFilledVal is not None:
            arrFilledValFrame = self.genArrValFrame(
                strFilledVal, arrRawFrame, ref=lsArrReferenceFrame)
            if arrFilledValFrame is None:
                res = self.invalidFrameShapeProcess(
                    mode=strInvalidFrameShapeProcessMode, message=message,
                    strRawFrameFilePath=strRawFrameFilePath)
                if not(res):
                    return None
        else:
            arrFilledValFrame = arrRawFrame
        if strValidPixelCondition is not None:
            arrIsValidPixelFrame = self.genArrValFrame(
                strValidPixelCondition, arrRawFrame, ref=lsArrReferenceFrame)
            if arrIsValidPixelFrame is None:
                res = self.invalidFrameShapeProcess(
                    mode=strInvalidFrameShapeProcessMode, message=message,
                    strRawFrameFilePath=strRawFrameFilePath)
                if not(res):
                    return None
        else:
            arrIsValidPixelFrame = np.ones(self.tpFrameShape, dtype=bool)
        if arrFilledValFrame.shape != arrIsValidPixelFrame.shape:
            res = self.invalidFrameShapeProcess(
                mode=strInvalidFrameShapeProcessMode, message=message,
                strRawFrameFilePath=strRawFrameFilePath)
            if not(res):
                return None
        if eventData is not None:
            arrIsValidPixelFrame *= ~eventData.genArrIsEventRangeFrame(
                arrRawFrame.shape)
            if excludeRim:
                arrIsValidPixelFrame[:2*eventData.maxLeak - 1, :] = False
                arrIsValidPixelFrame[-2*eventData.maxLeak + 1:, :] = False
                arrIsValidPixelFrame[:, :2*eventData.maxLeak - 1] = False
                arrIsValidPixelFrame[:, -2*eventData.maxLeak + 1:] = False
        if ignoreNan:
            arrIsValidPixelFrame *= ~np.isnan(arrFilledValFrame)
        if ignoreInf:
            arrIsValidPixelFrame *= arrFilledValFrame != np.inf
        if ignoreNinf:
            arrIsValidPixelFrame *= arrFilledValFrame != -np.inf
        arrValidFilledVal = arrFilledValFrame[arrIsValidPixelFrame]
        self.frameCnt += 1
        self.lsStrRawFrameFilePath.append(strRawFrameFilePath)
        self.arrCnt += np.histogram(arrValidFilledVal, bins=self.arrBins)[0]
    def genArrValFrame(self, strVal, raw, ref=[]):
        Y = (
            np.arange(self.tpFrameShape[0]).reshape((self.tpFrameShape[0], 1))
            * np.ones(self.tpFrameShape))
        X = (
            np.arange(self.tpFrameShape[1]).reshape((1, self.tpFrameShape[1]))
            * np.ones(self.tpFrameShape))
        try:
            arrRet = eval(strVal)
            return arrRet
        except:
            return None
    def genArrHistVal(self):
        arrRet = np.zeros((2, self.arrBins.size))
        arrRet[0][:-1] = self.arrCnt
        arrRet[1] = self.arrBins
        return arrRet
    def saveAsSpectrumBinFile(
            self, strFilePath, message=True, dicAppendixHeader={}):
        arrHistVal = self.genArrHistVal()
        dicHeader = {}
        for cnt, strRawFrameFilePath in enumerate(self.lsStrRawFrameFilePath):
            dicHeader['F' + str(cnt)] = strRawFrameFilePath
        dicHeader.update(dicAppendixHeader)
        saveAsFits(arrHistVal, strFilePath, message=message, header=dicHeader)


class ZScaleManager():
    def __init__(self, arrSource=None):
        self.arrSource = arrSource
        self.zScaleInterval = ZScaleInterval(nsamples=600)
        self.lsLimit = None
        self.imgZScale = None
    def setArrSource(self, arrSource):
        self.arrSource = arrSource
    def setLsLimit(self):
        arrLimit = self.zScaleInterval.get_limits(self.arrSource)
        self.lsLimit = [float(arrLimit[0]), float(arrLimit[1])]
    def setImgZScale(self):
        arrZScale = self.arrSource.copy()
        arrZScale[arrZScale <= self.lsLimit[0]] = self.lsLimit[0]
        arrZScale[self.lsLimit[1] <= arrZScale] = self.lsLimit[1]
        arrImgZScale = np.zeros(self.arrSource.shape+(3,), dtype='uint8')
        arrImgZScale[:, :, :3] = ((arrZScale[::-1] - self.lsLimit[0]) / (
                    self.lsLimit[1] - self.lsLimit[0]
            ) * (2**8 - 1)).astype('uint8').reshape(self.arrSource.shape+(1,))
        arrIsNan = np.isnan(self.arrSource)
        arrImgZScale[arrIsNan[::-1], 0] = 255
        arrImgZScale[arrIsNan[::-1], 1:3] = 0
        self.imgZScale = Image.fromarray(arrImgZScale, mode='RGB')
