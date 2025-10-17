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
import itertools

from collections import OrderedDict

import cv2
import numpy as np
import numpy.linalg
import tqdm

import ifs

H_BASE_MAX = 220
S_BASE_MAX = 200
V_BASE_MAX = 200

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
   print('singularMat::{}'.format(s))
   print('rightSingularVector::{}'.format(V))
   print('theta:{}'.format(math.atan2(U[1, 0], U[0, 0])*360.0/2.0/math.pi))
   print('phi:{}'.format(math.atan2(V[1, 0], V[0, 0])*360.0/2.0/math.pi))
   
   return affine



def conf():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_systems', type=str,   required = True)
    parser.add_argument('--num_perturbations', type=int,   default  = 10)
    parser.add_argument('--num_patch_aug', type=int,   default  = 1)
    parser.add_argument('--flip_aug',      type=str,   default  = 'False', choices=['True', 'False'])
    parser.add_argument('--color_aug',     type=str,   default  = 'False', choices=['True', 'False'])
    parser.add_argument('--save_dir',    type = str,  required=True)
    parser.add_argument('--numof_points',type = int,  default=100000)
    parser.add_argument('--img_size',    type = int,  default=362)
    parser.add_argument('--img_type',      type=str,   default  = 'binary', choices=['binary', 'gray'])
    parser.add_argument('--seed',        type = int,  default=42)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    #引数解釈
    args = conf()

    with open(args.saved_systems) as fp:
       classesParams = json.load(fp)
    
    img_type        = args.img_type
    img_type_binary = True  if args.img_type == 'binary'  else False
    flip_aug        = False if args.flip_aug == 'False'   else True
    color_aug       = False if args.color_aug == 'False'  else True
    
    save_dir          = args.save_dir
    numofPoints       = args.numof_points
    num_perturbations = args.num_perturbations
    num_patch_aug     = args.num_patch_aug
    img_size          = args.img_size

    #乱数初期化
    rng = np.random.default_rng(seed=args.seed)
    random.seed(args.seed)

    #辞書の用意：実験条件の保存
    instanceParamsDict = OrderedDict()
    instanceParamsDict['num_classes']   = classesParams['num_classes']
    #instanceParamsDict['num_instances'] = num_perturbations * num_patch_aug * (4 if flip_aug else 1) 
    instanceParamsDict['num_instances'] = num_perturbations * 3
    instanceParamsDict['numof_points']  = numofPoints
    instanceParamsDict['img_size']      = img_size
    instanceParamsDict['img_type']      = args.img_type
    instanceParamsDict['flip_aug']      = flip_aug
    instanceParamsDict['color_aug']      = color_aug
    instanceParamsDict['instances']     = OrderedDict()
    
    os.makedirs(save_dir,   exist_ok=True)
    
    perturbations = [ 
                      [ 0.00, 0.00,   0.00,  0.00,  0.00,  0.00 ], #00
                      
                      [ 0.05, 0.00,   0.00,  0.00,  0.00,  0.00 ], #01 sigL
                      [-0.05, 0.00,   0.00,  0.00,  0.00,  0.00 ], #02
                      
                      [ 0.00,  0.05,  0.00,  0.00,  0.00,  0.00 ], #03 sigS
                      [ 0.00, -0.05,  0.00,  0.00,  0.00,  0.00 ], #04 
                      
                      [ 0.00,  0.00,  0.10,  0.00,  0.00,  0.00 ], #05 theta
                      [ 0.00,  0.00, -0.10,  0.00,  0.00,  0.00 ], #06 

                      [ 0.00,  0.00,  0.00,  0.10,  0.00,  0.00 ], #07 phi
                      [ 0.00,  0.00,  0.00, -0.10,  0.00,  0.00 ], #08

                      [ 0.00,  0.00,  0.00,  0.00,  0.10,  0.00 ], #09 e
                      [ 0.00,  0.00,  0.00,  0.00, -0.10,  0.00 ], #10 f
                      
                      [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.10 ], #11
                      [ 0.00,  0.00,  0.00,  0.00,  0.00, -0.10 ], #12
                    ]
    
    excludes_perturb = [(1, 2), (1, 5), (1, 6), (2, 5), (2, 6), (3, 4), (3, 7), (3, 8),
                        (5, 6), (5, 7), (5, 8), (6, 7), (6, 8), (7, 8), (9, 10), (11, 12)]
    perturbIdcs = list(itertools.combinations(range(len(perturbations)), 2))
    perturbIdcs = [ pair for pair in perturbIdcs if pair not in excludes_perturb  ]

    #クラス毎Fractalパラメータの読み出し
    systems_dict = classesParams['classes']

    rots = [0, 2, 4, 6]
    
    for cls, params in tqdm.tqdm(systems_dict.items()):
    
        savedir_cls = os.path.join(save_dir, cls)
        os.makedirs(savedir_cls, exist_ok=True)
        
        spectral = params['spectral']
        
        samplePerturbIdcs = random.sample(perturbIdcs, k=num_perturbations)
        
        if color_aug:
            h_base = random.randint(0,   H_BASE_MAX)
            s_base = random.randint(77,  S_BASE_MAX)
            v_base = random.randint(128, V_BASE_MAX)
        
        for p in range(num_perturbations):
           instanceName     = cls + '_' + ('%02d' % p)
           fractalParams    = np.zeros( (len(spectral), 2, 3), dtype='float')
           instancdePerturb = np.array(perturbations[samplePerturbIdcs[p][0]]) + np.array(perturbations[samplePerturbIdcs[p][1]])
           system              = []
           perturbatedSpectral = []
           for i, spectalParam in enumerate(spectral):
               instanceSpectral       = np.array(spectalParam) + instancdePerturb
               sigMat                 = np.array([ [instanceSpectral[0], 0.0], [0.0, instanceSpectral[1]]])
               theta, phi, e, f       = instanceSpectral[2], instanceSpectral[3], instanceSpectral[4], instanceSpectral[5]
               affine                 = getAffineSVD((theta * math.pi/180.0), (phi * math.pi/180.0), sigMat)
               a, b, c, d             = affine.transpose().ravel()
               fractalParams[i, 0, :] = [a, c, e]
               fractalParams[i, 1, :] = [b, d, f]

               system.append([a, b, c, d, e, f])
               perturbatedSpectral.append(instanceSpectral.tolist())
               
           #ifsから画像生成
           points = ifs.iterate(fractalParams, numofPoints, rng_list=rng.random(numofPoints), ps=None)
           xyrange = ifs.minmax(points)
           region = np.concatenate(xyrange)
           for rot in rots:
              strRot = ('%02d' % rot)
              points = ifs.rotatePoints(points, (rot * 2.0 * math.pi/360.0), xyrange)
              gray_image = ifs.render(points, s=img_size, region=region, binary=True)
              gray_image = (gray_image/gray_image.max()).astype('uint8')
              if color_aug:
                  h = h_base + random.randint(0, (255 - H_BASE_MAX))
                  s = s_base + random.randint(0, (255 - S_BASE_MAX))
                  v = v_base + random.randint(0, (255 - V_BASE_MAX))
                  image = np.empty((img_size, img_size, 3), dtype=np.uint8)
                  image[:, :, 0] = gray_image * h
                  image[:, :, 1] = gray_image * s
                  image[:, :, 2] = gray_image * v
                  image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR_FULL)
              else:
                  image = gray_image * 128
              cv2.imwrite(os.path.join(savedir_cls, (instanceName + '_' + strRot +'.png')), image)
           
           instanceParamsDict['instances'][instanceName]             = OrderedDict()
           instanceParamsDict['instances'][instanceName]['system']   = system
           instanceParamsDict['instances'][instanceName]['spectral'] = spectral

    #生成されたパラメータ全部を辞書に保存
    with open(os.path.join(save_dir, os.path.basename(save_dir) + '.json'), 'w') as fp:
       from compactJSONEncoder import CompactJSONEncoder
       json.dump(instanceParamsDict, fp, cls=CompactJSONEncoder, indent=2)