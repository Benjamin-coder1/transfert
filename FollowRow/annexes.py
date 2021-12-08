# -*- coding: utf-8 -*-
"""rpmodules.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ih9ObD2L-Xsg-LYnoBUumqfpiy99l6ZL

# Setup
"""
import os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pickle
from numpy.lib.function_base import extract
from sklearn.linear_model import LinearRegression
from statistics import mode
import pyrealsense2 as rs
from scipy.ndimage import median_filter, generic_filter
from xml.dom.minidom import parse, parseString
import time 

point_clouds = '/media/ben/easystore/data_collection/sensor_captures/data/depth_sider_cam/matrices/'
src_dir = '/media/ben/easystore/data_collection/sensor_captures/data/rgb_sider_cam/' 
# _, axs = plt.subplots(nrows=1, ncols=2)


def compareTrajectory(newtraj, lasttraj) : 
    """ This function compare the two trajectories"""
    return (lasttraj - newtraj) / lasttraj


def add_ComplexeRectangle(d0, angle, ax, L, l) :
    x = L*np.sin(angle) + d0
    angle = angle*180/np.pi
    rect = patches.Rectangle((x,0), l, L, angle= angle, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

def add_rectangle(boxes, ax, im_w, im_h) :
    bbspl = boxes.split(' ')
    # extract data in YOLO format
    xc = float(bbspl[1])
    yc = float(bbspl[2])
    w = float(bbspl[3])
    h = float(bbspl[4])
    # calculated values
    x0 = math.ceil((xc - w / 2) * im_w)
    y0 = math.ceil((yc - h / 2) * im_h)
    x1 = math.floor((xc + w / 2) * im_w)
    y1 = math.floor((yc + h / 2) * im_h)
    
    # draw bounding boxes on image
    rect = patches.Rectangle((x0, y0), (x1 - x0), (y1 - y0), linewidth=1, edgecolor='r',
                                facecolor='none')    
    ax.add_patch(rect)

def putDepthTreshold(pc, treshold, v=2) : 
    depth = []
    for i in range(480*640) : 
        if pc[i,v] > treshold[1] or pc[i,v] < treshold[0] :          
            depth.append(0)
        else :        
            depth.append(pc[i,v]) 
    depth  = np.array(depth )
    return np.reshape(depth,  (480, 640))

def computeXZ(boxes, counter = 1285, threshold=(0.7,1.3)):      

    img = src_dir + 'side_rgb_2021-01-01-0123456789_' + str(counter) + '.jpg'     
    # create histograms of pixel depth for each bounding box 
    im = Image.open(os.path.join(src_dir, img))
    axs[0].imshow(im)
    im_w, im_h = im.size

    trunk_dists, trunk_ys, trunk_closest_pts= 0, 0, 0
   
    # load data from point cloud pickle
    with open(point_clouds + 'side_point_cloud_2021-01-01-0123456789_' + str(counter) + '.pkl', 'rb') as pcp:
        deb = time.time()
        p_c = pickle.load(pcp)
        point_cloud = np.reshape(p_c, (im_h, im_w, 3))        

        # print(f'bounding box {i+1}')
        bbspl = boxes.split(' ')
        # extract data in YOLO format
        xc = float(bbspl[1])
        yc = float(bbspl[2])
        w = float(bbspl[3])
        h = float(bbspl[4])
        # calculated values
        x0 = math.ceil((xc - w / 2) * im_w)
        y0 = math.ceil((yc - h / 2) * im_h)
        x1 = math.floor((xc + w / 2) * im_w)
        y1 = math.floor((yc + h / 2) * im_h)
     
        # draw bounding boxes on image
        add_rectangle(boxes, axs[0], im_w, im_h)

        hist_data = (point_cloud[y0:y1, x0:x1, 2]).flatten()
        n, bins, _ = axs[1].hist(hist_data, bins=100, range=[0, 4], histtype='step')
       
        # pick peak in range 0.5 - 1.5
        start_ind = 0
        end_ind = 0
        for ind in range(len(bins)):
            if round(bins[ind], 1) <= threshold[0]:
                start_ind = ind
            elif round(bins[ind], 1) >= threshold[1]:
                end_ind = ind
                break

        max_ind = np.argmax(np.array([n[ind] for ind in range(start_ind, end_ind)])) + start_ind
        try:
            trunk_dists = mode(
                [i for i in hist_data if (i >= bins[max_ind]) and (i <= bins[max_ind + 1])])
        except:
            trunk_dists = 0.5  # ?

        trunk_ys = (y0 + y1) // 2

        # find point with closest depth to average trunk depth
        depth = 0
        closest = (0, 0)
        for y in range(y0, y1):
            for x in range(x0, x1):
                if abs(point_cloud[y][x][2] - trunk_dists) < abs(
                        point_cloud[closest[0]][closest[1]][2] - trunk_dists):
                    closest = (y, x)
                    depth = abs(point_cloud[y][x][2] - trunk_dists)
        # print(depth)
        trunk_closest_pts = closest

        # linear regression
        x = point_cloud[trunk_closest_pts[0]][trunk_closest_pts[1]][0]
        z = trunk_dists
        # model = LinearRegression().fit(x, z)
        # print(f'r squared {round(model.score(x,z), 3)} intercept {round(model.intercept_, 3)} slope {round(model.coef_[0], 3)}')

    # plt.show()
    return z, x
    
                 

