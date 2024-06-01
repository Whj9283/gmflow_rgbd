import cv2 as cv
import numpy as np
import re
import os
import os.path as osp
from glob import glob

def ReadPFMFile(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)      # 垂直翻转
    return data.astype(np.float32)


def depthWrite(filename, depth):
    """ Write depth to file. """
    height, width = depth.shape[:2]
    f = open(filename, 'wb')
    # write the header
    f.write('PIEH')
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)

    depth.astype(np.float32).tofile(f)
    f.close()


def Disparity2Depth() -> None:
    prefixPath = "F:\\datasets\\Monkaa\\disparity\\disparity"
    prefixSavePath = "F:\\datasets\\Monkaa\\dpeth\\depth"
    dirList = os.listdir(prefixPath)
    for scene in dirList:
        scenePath = osp.join(prefixPath, scene, 'left')
        disparities = sorted(glob(osp.join(scenePath, '*.pfm')))
        for disparityPath in disparities:
            disparitiesFileName = os.path.basename(disparityPath)
            FileName = os.path.splitext(disparitiesFileName)[0]
            sceneSavePath = osp.join(prefixSavePath, scene, 'left')
            savePath = osp.join(sceneSavePath, FileName + '.dpt')
            depth = 1050.0 / ReadPFMFile(disparityPath)
            depthWrite(savePath, depth)


    # disparity = ReadPFMFile(disparityPath)
    # depth = 1050.0 / disparity
    # depth = cv.normalize(depth, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    # disparity = cv.normalize(disparity, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    # cv.imshow("Disparity", disparity)
    # cv.imshow("Depth", depth)

    cv.waitKey(10000)

Disparity2Depth()
