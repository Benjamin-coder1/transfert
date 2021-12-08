import numpy as np 
import matplotlib.pyplot as plt 
import pickle,annexes, math
import scipy.stats as stats
import csv, time
import trajectorySVG


PATH = '/media/ben/easystore/data_collection/sensor_captures/data/'
counter = 1300
boxes = "0 0.8 0 0.3 2"
x0, l0, L, l, v0 = 2, 1, 0.4, 0.44, 1
treshold1, treshold2 = (0.7, 1.3), (0.7, 1.6)
# pc = pickle.load(open(PATH + 'depth_sider_cam/matrices/side_point_cloud_2021-01-01-0123456789_' + str(counter) + '.pkl', 'rb')) 
# imgS = plt.imread(open(PATH + 'rgb_sider_cam/side_rgb_2021-01-01-0123456789_' + str(counter) + '.jpg', 'rb')) 
# imgF = plt.imread(open(PATH + 'rgb_front_cam/front_rgb_2021-01-01-0123456789_' + str(counter) + '.jpg', 'rb'))
# im_h, im_w = np.shape(imgS)[0], np.shape(imgS)[1]



#######  ----- DISPLAY  ------ #########
# _, axs = plt.subplots(nrows=2, ncols=3)
# axs = axs.flatten()

# axs[0].imshow(imgS )
# axs[0].set_title('Rgb side')
# annexes.add_rectangle(boxes, axs[0], im_w, im_h)

# axs[1].set_title('Rgb front')
# axs[1].imshow(imgF )

# axs[2].set_title('Depth side')
# axs[2].imshow( annexes.putDepthTreshold(pc, (0,5)) )
# annexes.add_rectangle(boxes, axs[2], im_w, im_h)

# axs[3].set_title('Depth side [0.7, 1.3]')
# axs[3].imshow( annexes.putDepthTreshold(pc, treshold1) )
# annexes.add_rectangle(boxes, axs[3], im_w, im_h)

# axs[4].set_title('Depth side [0.7, 1.6]')
# axs[4].imshow( annexes.putDepthTreshold(pc, treshold2) )
# annexes.add_rectangle(boxes, axs[4], im_w, im_h)

# d0,_ = annexes.computeXZ(boxes, counter=counter, threshold=treshold1)
# angle = 1e-1
# annexes.add_ComplexeRectangle(x0, d0, angle, axs[5], L, l) 
# annexes.computeTrajectory(speed, d0, angle, 15, axs[5], L, l, x0, 3, 1) 
# axs[5].scatter( [x0]*10, np.linspace(-5,5, 10))
# axs[5].scatter( [x0 + l0]*10, np.linspace(-5,5, 10))
# axs[5].scatter( [x0 + l0/2]*15, np.linspace(-5,10, 15), marker='|', color='grey', alpha = 1/2) 
# axs[5].set_title('depth : ' + str(d0) + 'm')
# plt.show()


#######  ----- CSV  ------ #########
# file = open('CSV/measure.csv', 'w', newline='') 
# writer = csv.writer(file, delimiter=';')
# writer.writerow(["image number" , 
#                     "Distance (R)", 
#                     "Z score-3 (R)", 
#                     "Z score-5 (R)", 
#                     "Z score-7 (R)",
#                     "Probability (R)",
#                     "Distance (L)", 
#                     "Z score-3 (L)", 
#                     "Z score-5 (L)", 
#                     "Z score-7 (L)",
#                     "Probability (L)",
#                     "Orientation", 
#                     "Z score-3 (F)", 
#                     "Z score-5 (F)", 
#                     "Z score-7 (F)",
#                     "Probability (F)" ])

# counterMin, counterMax = 1250,1270
# x0, l0, L, l,speed = 2, 3, 3.5, 1.5, 1
# lstSvg = [0]*10
# for counter in range(counterMin, counterMax) : 
#     ## Image processing ##
#     d0,_ = annexes.computeXZ(boxes, counter=counter, threshold=treshold1)
#     lstSvg = [d0] + lstSvg[:-1]
#     zScore3R = stats.zscore( lstSvg[:3])[0]
#     zScore5R = stats.zscore( lstSvg[:5])[0]
#     zScore7R = stats.zscore( lstSvg[:7])[0]
#     probaR = 4

#     ## value writting ##
#     writer.writerow([counter,           # Image Number 
#                     d0,                 # Distance (R)
#                     zScore3R,           # Z score-3 (R)
#                     zScore5R,           # Z score-5 (R)
#                     zScore7R,           # Z score-7 (R)
#                     probaR,             # probability (R)
#                     0,                  # Distance (L)
#                     0,                  # Z score-3 (L)
#                     0,                  # Z score-5 (R)
#                     0,                  # Z score-7 (R)
#                     0,                  # probability (L)
#                     0,                  # orientation
#                     0,                  # Z score-3 (F)
#                     0,                  # Z score-5 (L)
#                     0,                  # Z score-7 (L)
#                     0 ])                # probability (F)

##### ----- TRAJECTORY ------ #######

trajectorySVG.followRow(v0, vehicle=(L,l,l0) , nbTemps = 10, K = 1)               


