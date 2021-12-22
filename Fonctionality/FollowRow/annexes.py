import os, math, time
from statistics import mode
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pyrealsense2 as rs
 

def compareTrajectory(newtraj, lasttraj) : 
    """ This function compare the two trajectories"""
    return (lasttraj - newtraj) / lasttraj

def add_ComplexeRectangle(d0, angle, ax, L, l) :
    x = L*np.sin(angle) + d0
    angle = angle*180/np.pi
    rect = patches.Rectangle((x,0), l, L, angle= angle, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

def add_rectangle(boxes, ax, im_w, im_h) :
    # print(f'bounding box {i+1}')
    bbspl = boxes.split(' ')
    # extract data in YOLO format
    x0 = math.ceil(float(bbspl[0]))
    y0 = math.ceil(float(bbspl[1]))
    x1 = math.ceil(float(bbspl[2]))
    y1 = math.ceil(float(bbspl[3]))
    
    # draw bounding boxes on image
    rect = patches.Rectangle((x0, y0), (x1 - x0), (y1 - y0), linewidth=1, edgecolor='r',
                                facecolor='none')    
    ax.add_patch(rect)


def computeXZ(boxes, pipeline, axs , threshold, logger=False):
    frames = pipeline.wait_for_frames()
    
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    im = np.asanyarray(rs.colorizer().colorize(depth_frame).get_data())
                
    # create histograms of pixel depth for each bounding box 
    imD = np.asanyarray(color_frame.get_data())
    
    axs[0].imshow(np.concatenate((im, imD), axis=0) )
    im_w, im_h = np.shape(im)[:2]
    trunk_dists, trunk_ys, trunk_closest_pts= 0, 0, 0
    
    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)
    v = points.get_vertices()
    p_c = np.asanyarray(v).view(np.float32).reshape(-1,3)
   
    # load data from point cloud pickle
    point_cloud = np.reshape(p_c, (im_w, im_h, 3))
    
    # print(f'bounding box {i+1}')
    bbspl = boxes.split(' ')
    # extract data in YOLO format
    x0 = math.ceil(float(bbspl[0]))
    y0 = math.ceil(float(bbspl[1]))
    x1 = math.ceil(float(bbspl[2]))
    y1 = math.ceil(float(bbspl[3]))       
 
    # draw bounding boxes on image
    add_rectangle(boxes, axs[0], im_w, im_h) 
    hist_data = (point_cloud[y0:y1, x0:x1, 2]).flatten()
    n, bins, _ = axs[1].hist(hist_data, bins=100, range=[0, 4], histtype='step')
   
    # pick peak in range 0.5 - 1.5
    start_ind = 0
    end_ind = 0
    for ind in range(len(bins)):
        if round(bins[ind], 1) <= threshold[0]: start_ind = ind
        elif round(bins[ind], 1) >= threshold[1]:
            end_ind = ind
            break

    max_ind = np.argmax(np.array([n[ind] for ind in range(start_ind, end_ind)])) + start_ind
    try: trunk_dists = mode([i for i in hist_data if (i >= bins[max_ind]) and (i <= bins[max_ind + 1])])
    except: trunk_dists = 0.5  # ?

    # find point with closest depth to average trunk depth
    depth = abs(point_cloud[y0:y1, x0:x1, 2] - trunk_dists)
    closest = np.array(np.unravel_index(np.argmin(depth, axis=None), depth.shape))
    closest[0] += y0
    closest[1] += x0
                
    # print(depth)
    circ = patches.Circle( [closest[1], closest[0] ] ,radius=10, edgecolor='k', fill=True)
    axs[0].add_patch(circ)

    # Coordinate 
    x = point_cloud[closest[0]][closest[1]][1]
    z = trunk_dists
    return z, x
  






