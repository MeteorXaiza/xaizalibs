# coding:utf-8


from PIL import Image

from .standardlib import *


########################################
###########PIL関連#######################
########################################
def getImg(strFilePath, message=False):
    if message:
        print('loading ' + strFilePath + '...')
    ret = Image.open(strFilePath)
    if message:
        print('finished.')
    return ret


def saveAsImg(img, strFilePath, message=False):
    img.save(strFilePath)
    if message:
        print(strFilePath + ' has been saved.')
