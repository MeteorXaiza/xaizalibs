# coding:utf-8


from math import *
import configparser
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request


def cp(strInputFilePath, strOutputFilePath, dir=True, message=True, cp2=False):
    if dir:
        strOutputDirPath = genLsStrDirPathAndFileName(strOutputFilePath)[0]
        if not(os.path.isdir(strOutputDirPath)):
            mkdirs(strOutputDirPath, message=message)
    if message:
        print((
            'copying ' + strInputFilePath
            + ' to ' + strOutputFilePath + '...'))
    if cp2:
        shutil.copy2(strInputFilePath, strOutputFilePath)
    else:
        shutil.copy(strInputFilePath, strOutputFilePath)
    if message:
        print('finished.')


def command(strCmd):
    subprocess.call(shlex.split(strCmd))


def download(strURL, strFilePath, message=False):
    if message:
        print('downloading ' + strURL + '...')
    urllib.request.urlretrieve(strURL,"{0}".format(strFilePath))
    # urllib.request.URLopener.retrieve(strURL, "{0}".format(strFilePath))
    if message:
        print('finished.')


def genDeepSizeOf(arg):
    ret = 0
    ret += sys.getsizeof(arg)
    if type(arg) in [list, tuple]:
        for cnt in range(len(arg)):
            ret += genDeepSizeOf(arg[cnt])
    elif type(arg) == dict:
        for key in arg.keys():
            ret += genDeepSizeOf(key)
            ret += genDeepSizeOf(arg[key])
    return ret


def genDicLocalTime(fTime):
    tpStrWeek = (
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
        'Sunday')
    tpRet = tuple(time.localtime(fTime))
    tpStrKey = (
        'year', 'month', 'day', 'hour', 'min', 'sec', 'week_day', 'year_day',
        'isdst')
    dicRet = {}
    for cnt in range(len(tpStrKey)):
        dicRet[tpStrKey[cnt]] = tpRet[cnt]
    dicRet['week'] = tpStrWeek[tpRet[6]]
    return dicRet


def genEpocTime(arg):
    typeArg = type(arg)
    if typeArg == dict:
        tpStrKey = (
            'year', 'month', 'day', 'hour', 'min', 'sec', 'week_day',
            'year_day', 'isdst')
        lsTime = []
        for strKey in tpStrKey:
            lsTime.append(arg[strKey])
        return time.mktime(tuple(lsTime))
    elif typeArg == time.struct_time:
        return time.mktime(arg)


def genLsSortedFromLsKey(lsSorted, lsKey=None, lsReverse=None):
    if lsKey is not None:
        if type(lsKey[0]) == list:
            if len(lsKey) <= 1:
                return genLsSortedFromLsKey(lsSorted, lsKey[0], lsReverse)
            else:
                if lsReverse is None or lsReverse == False:
                    lsReverse = [False] * len(lsKey)
                elif lsReverse == True:
                    lsReverse = [True] * len(lsKey)
                return genLsSortedFromLsKey(
                    genLsSortedFromLsKey(
                        lsSorted, lsKey[-1], lsReverse[-1]),
                    lsKey[:-1], lsReverse[:-1])
        else:
            lsSortedKey = sorted(lsKey)
            sortedCnt, lsRet = 0, []
            while True:
                sortedKey = lsSortedKey[sortedCnt]
                for cnt, key in enumerate(lsKey):
                    if key == sortedKey:
                        lsRet.append(lsSorted[cnt])
                        sortedCnt += 1
                if sortedCnt >= len(lsSorted):
                    break
    else:
        lsRet = sorted(lsSorted)
    if lsReverse:
        return lsRet[::-1]
    else:
        return lsRet


def genLsStrDirPathAndFileName(strFilePath):
    match = re.match('(.*/)*(.*(\..*)?)', strFilePath)
    strDirPath, strFileName = match.groups()[:2]
    if strDirPath is None:
        strDirPath = ''
    return [strDirPath, strFileName]


def getConfig(arg, lsStrArgName, dicConfig, strKey1, strKey2, stringNone=True):
    ret = arg
    if not(any(getLsInLs(lsStrArgName, sys.argv))) and dicConfig is not None:
        if strKey1 in dicConfig.keys():
            if strKey2 in dicConfig[strKey1].keys():
                ret = dicConfig[strKey1][strKey2]
    if stringNone and ret == 'None':
        return None
    else:
        return ret


def getLsStrFileName(strDirPath, match=None):
    if strDirPath[-1:] != '/':
        strDirPath += '/'
    lsRet = []
    for strFileName in os.listdir(strDirPath):
        if os.path.isfile(strDirPath + strFileName):
            if match is None:
                lsRet.append(str(strFileName))
            else:
                if re.match(match, strFileName) is not None:
                    lsRet.append(str(strFileName))
    return lsRet


def getLsStrDirName(strDirPath, match=None):
    if strDirPath[-1:] != '/':
        strDirPath += '/'
    lsRet = []
    for strDirName in os.listdir(strDirPath):
        if os.path.isdir(strDirPath + strDirName):
            if match is None:
                lsRet.append(str(strDirName))
            else:
                if re.match(match, strDirName) is not None:
                    lsRet.append(str(strDirName))
    return lsRet


def getLsInLs(ls1, ls2):
    lsRet = [ls1[cnt] in ls2 for cnt in range(len(ls1))]
    return lsRet


def getMtime(strFilePath):
    return os.path.getmtime(strFilePath)


def getStrAbsPath(strPath, slash=True, windows=True):
    strRet = os.path.abspath(strPath)
    if slash:
        # 存在していて、ディレクトリ
        if os.path.isdir(strRet):
            strRet += '/'
        # 存在していないが、ディレクトリ（末尾が/または\\）
        elif strPath[-1] in ['/', '\\']:
            strRet += strPath[-1]
    if os.path.isdir(strRet):
        if not(slash) and strRet[-1] == '/':
            strRet = strRet[:-1]
    if windows:
        strRet = strRet.replace('\\', '/')
    return strRet


def getStrSelect(strMessage=None, lsStrValid=None):
    if strMessage is not None:
        print(strMessage)
    if lsStrValid is not None:
        while True:
            strInput = input()
            if strInput in lsStrValid:
                break
    else:
        strInput = input()
    return strInput


def mkdirs(strDirPath, message=False):
    lsStrDirName = strDirPath.split('/')
    strTargetDirPath = ''
    for cnt, strDirName in enumerate(lsStrDirName):
        strTargetDirPath += strDirName + '/'
        if not os.path.isdir(strTargetDirPath):
            if message:
                print('making ' + strTargetDirPath + '...')
            os.mkdir(strTargetDirPath)
            if message:
                print('finished.')


def prints(*lsMessage, head=''):
    if head == '':
        for message in lsMessage:
            print(message)
    else:
        for message in lsMessage:
            print(head + str(message).replace('\n', '\n'+head))


########################################
###########iniファイル関連################
########################################
def getDicIni(strFilePath, encoding='utf-8', message=False):
    iniObj = configparser.ConfigParser()
    iniObj.optionxform = str
    if message:
        print('loading ' + strFilePath + '...')
    iniObj.read(strFilePath, encoding=encoding)
    if message:
        print('finished.')
    lsSections = iniObj.sections()
    dicRet = {}
    for section in lsSections:
        dicRet[str(section)] = {}
        lsOptions = iniObj.options(section)
        for option in lsOptions:
            dicRet[str(section)][str(option)] = str(iniObj.get(section, option))
    return dicRet


def saveAsIni(dic, strFilePath, message=False):
    lsStrTxtLine = []
    lsSections = dic.keys()
    for section in lsSections:
        lsOptions = dic[section].keys()
        lsStrTxtLine.append('[' + str(section) + ']')
        for option in lsOptions:
            lsStrTxtLine.append(str(option) + ' = ' + str(dic[section][option]))
    saveAsTxt(lsStrTxtLine, strFilePath, message)


########################################
###########JSONファイル関連################
########################################
def getDicJSON(strFilePath, message=False):
    lsStrTxtLine = getLsStrTxtLine(strFilePath, message=message)
    return json.loads(''.join(lsStrTxtLine))


def saveAsJSON(dic, strFilePath, indent=None, message=False):
    with open(strFilePath, 'w') as f:
        json.dump(dic, f, indent=indent)
    if message:
        print(strFilePath + ' has been saved.')


########################################
###########TXTファイル関連################
########################################
def getLsStrTxtLine(strFilePath, encoding='utf-8', message=False):
    if message:
        print('loading ' + strFilePath + '...')
    with open(strFilePath, encoding=encoding) as f:
        if message:
            print('finished.')
        lsRet = [s.strip('\n') for s in f.readlines()]
        return lsRet


def saveAsTxt(ls, strFilePath, message=False, encoding='utf-8'):
    with open(strFilePath, mode='w', encoding='utf-8') as f:
        f.write('\n'.join(ls))
    if message:
        print(strFilePath + " has been saved.")
