# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 23:55:12 2017

@author: Kazushige Okayasu, Hirokatsu Kataoka
"""

import os
import argparse
import json
import random
import math

from collections import OrderedDict

import cv2
import numpy as np
import numpy.linalg

#a > bを保証する
def sortPair(a, b):
   tmp = a
   if b > a:
      a = b
      b = tmp
   return a, b
   
def getAffineSVD(theta, phi, singMat):
   leftMat  = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
   rightMat = np.array([[math.cos(phi),   -math.sin(phi)],   [math.sin(phi),   math.cos(phi)]])
   affine = leftMat.dot(singMat).dot(rightMat.transpose())
   
   
   U, s, V = np.linalg.svd(affine, full_matrices=True)
   print('affine::{}'.format(affine))
   print('leftSingularVector::{}'.format(U))
   print('rightSingularVector::{}'.format(V))
   print('theta:{}'.format(math.atan2(U[1, 0], U[0, 0])*360/2.0/math.pi))
   print('phi:{}'.format(math.atan2(V[1, 0], V[0, 0])*360/2.0/math.pi))
   
   return affine

import ifs

sig1_default = [1.0, 0.9, 0.8, 0.7]

sig2_default = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

#thetas_default = [0.0, 40.0, 80.0, 120.0, 160.0]
thetas_default = [-144.0, -120.0, -96.0, -72.0, -48.0, -24.0, 0.0, 24.0, 48.0, 72.0, 96.0, 120.0, 144.0]

ef_default = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

#sigList = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
sigList = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


used = []

def sample_systemGrid(affineNum=4, targetSigma=4.7, sigList=sigList, thetas=thetas_default, phis=None, ef=ef_default):
        
        if phis == None:
            phis = thetas

        systemParams  = []
        spectalParams = []
        
        singList  = []
        thetaList = []
        phiList   = []
        eList     = []
        fList     = []
        
        while True:
           sampledSigs = random.sample(sigList, k=(affineNum * 2))
           sigmaFactor = 0.0
           for i in range(affineNum):
              l, s = sampledSigs[i*2], sampledSigs[i*2 + 1]
              l, s = sortPair(l, s)
              singList.append(l)
              singList.append(s)
              sigmaFactor = sigmaFactor + l + 2.0 * s

           if sampledSigs in used:
               singList  = []
               continue
           elif sigmaFactor !=  targetSigma:
               singList  = []
               continue
           else:
               used.append(sampledSigs)
               thetaList = random.sample(thetas,     k=affineNum)
               phiList   = random.sample(phis,       k=affineNum)
               eList     = random.sample(ef_default, k=affineNum)
               fList     = random.sample(ef_default, k=affineNum)
               break

        for i in range(affineNum):
           singL, singS = singList[i*2], singList[i*2 + 1]
           theta = thetaList[i]
           phi   = phiList[i]
           e     = eList[i]
           f     = fList[i]

           singMat    = np.array([[singL, 0.0], [0.0, singS]])
           affine     = getAffineSVD(theta, phi, singMat)
           a, b, c, d = affine.transpose().ravel()
           p          = abs(singL * singS)
           systemParams.append([a, b, c, d, e, f, p])
           spectalParams.append([singL, singS, theta, phi, e, f])
        
        
        systemParams  = sorted(systemParams,  key=lambda x:x[-1],     reverse=True)
        spectalParams = sorted(spectalParams, key=lambda x:x[0]*x[1], reverse=True)
        
        return systemParams, spectalParams, sigmaFactor



def conf():
	parser = argparse.ArgumentParser()
	parser.add_argument('--save_dir',    type = str,  required=True)
	parser.add_argument('--num_systems', type = int,  default=1000)
	parser.add_argument('--num_affine',  type = int,  default=4)
	parser.add_argument('--numof_points',type = int,  default=100000)
	parser.add_argument('--img_size',    type = int,  default=362)
	parser.add_argument('--sigma_factor',type = float,default=4.7)
	parser.add_argument('--fillrate_min',type = float,default=0.05)
	parser.add_argument('--fillrate_max',type = float,default=1.0)
	parser.add_argument('--offset_cid',  type = int,  default=0)
	args = parser.parse_args()
	return args


if __name__ == "__main__":

    #乱数初期化
    rng = np.random.default_rng(seed=42)
    random.seed(42)
    
    #引数解釈
    args = conf()
    
    save_dir     = args.save_dir
    numofPoints  = args.numof_points
    num_systems  = args.num_systems
    num_affine   = args.num_affine
    img_size     = args.img_size
    sigma_factor = args.sigma_factor
    fillrateMin  = args.fillrate_min
    fillrateMax  = args.fillrate_max
    offset_cid   = args.offset_cid

    #辞書の用意：実験条件の保存
    fractalParamsDict = OrderedDict()
    fractalParamsDict['num_classes']  = num_systems
    fractalParamsDict['num_affine']   = num_affine
    fractalParamsDict['numof_points'] = numofPoints
    fractalParamsDict['img_size']     = img_size
    fractalParamsDict['sigma_factor'] = sigma_factor
    fractalParamsDict['fillrateMin']  = fillrateMin
    fractalParamsDict['fillrateMax']  = fillrateMax
    fractalParamsDict['classes']      = OrderedDict()

    savedir_img = os.path.join(save_dir, 'classImgs')
    os.makedirs(savedir_img,   exist_ok=True)
    
    
    fillrateMin = int(fillrateMin*img_size*img_size)
    fillrateMax = int(fillrateMax*img_size*img_size)
    
    i = 0
    while(i < num_systems):
        class_str = '%05d' % (i + offset_cid)
        sys, spectral, sfactor = sample_systemGrid(affineNum=num_affine, targetSigma=sigma_factor, sigList=sigList, thetas=thetas_default, phis=None, ef=ef_default)
        print('sys:{}'.format(sys))
        fractalParams = np.zeros((len(sys), 2, 3), dtype=float)
        for j , param in enumerate(sys):
            fractalParams[j, 0, :] = [param[0], param[2], param[4]]
            fractalParams[j, 1, :] = [param[1], param[3], param[5]]
        
        #ifsから画像生成
        points     = ifs.iterate(fractalParams, numofPoints, rng_list=rng.random(numofPoints), ps=None)
        gray_image = ifs.render(points, s=img_size, binary=True)
        
        nonzero = np.count_nonzero(gray_image)
        if (nonzero < fillrateMin) or  (fillrateMax < nonzero): #一定以上の占有率がなければ却下
            continue
        gray_image = ((gray_image/gray_image.max())*128).astype('uint8')
        cv2.imwrite(os.path.join(savedir_img, (class_str + '.png')), gray_image)
        
        fractalParamsDict['classes'][class_str]                = OrderedDict()
        fractalParamsDict['classes'][class_str]['system']      = sys
        fractalParamsDict['classes'][class_str]['spectral']    = spectral
        fractalParamsDict['classes'][class_str]['sigmaFactor'] = sfactor
        i = i + 1
        
    #生成されたパラメータ全部を辞書に保存
    with open(os.path.join(save_dir, os.path.basename(save_dir) + '.json'), 'w') as fp:
       from compactJSONEncoder import CompactJSONEncoder
       json.dump(fractalParamsDict, fp, cls=CompactJSONEncoder, indent=2)