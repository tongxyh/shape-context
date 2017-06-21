# coding=utf-8
import cv2
import sys
import numpy as np
import copy
from munkres import Munkres,print_matrix
#from matplotlib import pyplot as plt

def canny_points(edges, points_num=100):
    h, w = edges.shape
    #print(h, w)

    count = 0
    edges_sample = np.zeros((h, w))
    points = []
    while count < points_num:
        axis_h = np.random.randint(h, size=1)
        axis_w = np.random.randint(w, size=1)
        # print(axis_h,axis_w)
        # print(edgesA[axis_h[0],axis_w[0]])
        if edges[axis_h[0], axis_w[0]] > 1:
            ax = [axis_h[0], axis_w[0]]
            edges[axis_h[0]-3:axis_h[0]+3, axis_w[0]-3:axis_w[0]+3] = 0
            edges_sample[axis_h[0], axis_w[0]] = 255
            points.append(ax)
            count = count + 1
            #print(count)

    #cv2.imshow('canny edges', edges_sample)
    #cv2.waitKey(1)
    return points
def shape_bins(points):
    N = len(points)
    bins_all = []
    ang_Block = 12
    dis_Block = 5
    for point_o in points[:]:
        distances = []
        angle = []
        for point in points[:]:
            distance = np.sqrt((point_o[0] - point[0]) ** 2 + (point_o[1] - point[1]) ** 2)
            if distance > 0.00001:
                distances.append(distance)
                angl = np.arcsin((point_o[0] - point[0]) / distance)
                if point_o[1] - point[1] < 0 and point_o[0] - point[0] > 0:
                    angl = angl + pi / 2
                if point_o[1] - point[1] < 0 and point_o[0] - point[0] < 0:
                    angl = angl - pi / 2
                if angl < 0:
                    angl = 2 * pi + angl
                angle.append(np.floor(6.0 * angl / pi))  # sin
                # print(distance,angl)
        mean_dist = np.mean(distances)
        distances = distances / mean_dist

        # print(angle)
        # print(mean_dist)
        # print(distances)
        block_lens = 1
        distances_log = np.log(distances / block_lens)

        for x in range(len(distances_log)):
            if distances_log[x] <= 0:
                distances_log[x] = 0
            elif distances_log[x] <= 1:
                distances_log[x] = 1
            elif distances_log[x] <= 2:
                distances_log[x] = 2
            elif distances_log[x] <= 3:
                distances_log[x] = 3
            elif distances_log[x] <= 4:
                distances_log[x] = 4

        bins = np.zeros((dis_Block, ang_Block))
        for x in range(len(distances_log)):
            bins[int(distances_log[x]), int(angle[x])] = bins[int(distances_log[x]), int(angle[x])] + 1

        # np.arcsin
        # print(bins)
        # plt.imsave('xx%d.jpg'%point_o[0],bins)
        # plt.show()
        bins = np.reshape(bins,[ang_Block*dis_Block])
        bins_all.append(bins)
    return bins_all

def cost_matrix(bins_A,bins_B):
    row = 0
    col = 0
    cost = np.zeros((len(bins_A), len(bins_B)))
    for bin_A in bins_A:
        col = 0
        for bin_B in bins_B:
            # print(bin_A+bin_B)
            cost[row, col] = 0.5 * np.sum(((bin_A - bin_B) ** 2) / (bin_A + bin_B + 0.00000001))
            col = col + 1
        row = row + 1

        # cv2.imshow('xxx2',cost/255.0)
        # cv2.waitKey()
    return cost

pi = 3.1415926535

imageA_path = sys.argv[1]
#imageB_path = 'back_2.png'
imageB_path = sys.argv[2]

# read images A and B
imageA = cv2.imread(imageA_path)
imageB = cv2.imread(imageB_path)
imageA = cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
imageB = cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY)

# canny
minVal_canny = 100
maxVal_canny = 200
edgesA = cv2.Canny(imageA,minVal_canny,maxVal_canny)
edgesB = cv2.Canny(imageB,minVal_canny,maxVal_canny)
#cv2.imshow('edges',edgesA)
#cv2.waitKey()

# Randomly select some points
pointsA = canny_points(edgesA,int(sys.argv[3]))
pointsB = canny_points(edgesB,int(sys.argv[3]))

# Calculate shape context
# rotation invariance is not considered yet
bins_A = np.array(shape_bins(pointsA))
bins_B = np.array(shape_bins(pointsB))

# Calculate the cost matrix between two bins
cost = cost_matrix(bins_A,bins_B)
cost = cost.tolist()
#cost = [[50,61,23,98],[57,24,54,19],[78,73,7,46],[6,86,1,88]]
m = Munkres()
indexes = m.compute(cost)
#print_matrix('Lowest cost through this matrix:', cost)
total = 0
for row, column in indexes:
    value = cost[row][column]
    total += value
    #print('(%d, %d) -> %d' % (row, column, value))
print('total cost: %d' % total)

if total > 750:
    print('Same shape!')
else:
    print('Not the Same shape')

