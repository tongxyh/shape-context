# coding=utf-8
import cv2
import numpy as np
import copy
import munkres

def canny_points(edges):
    h, w = edges.shape
    #print(h, w)

    count = 0
    edges_sample = np.zeros((h, w))
    points = []
    while count < 200:
        axis_h = np.random.randint(h, size=1)
        axis_w = np.random.randint(w, size=1)
        # print(axis_h,axis_w)
        # print(edgesA[axis_h[0],axis_w[0]])
        if edges[axis_h[0], axis_w[0]] > 1:
            ax = [axis_h[0], axis_w[0]]
            edges[axis_h[0], axis_w[0]] = 0
            edges_sample[axis_h[0], axis_w[0]] = 255
            points.append(ax)
            count = count + 1
            #print(count)

    cv2.imshow('canny edges', edges_sample)
    cv2.waitKey(1)
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
        block_lens = 0.5
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

imageA_path = 'B.png'
imageB_path = 'BM.png'

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
cv2.imshow('xxx',edgesA)
cv2.waitKey(1)

# Randomly select some points
pointsA = canny_points(edgesA)
pointsB = canny_points(edgesB)

# Calculate shape context
# rotation invariance is not considered yet
bins_A = np.array(shape_bins(pointsA))
bins_B = np.array(shape_bins(pointsB))

# Calculate the cost matrix between two bins
#cost = cost_matrix(bins_A,bins_B)

cost = np.array([[50,61,23,98],[57,24,54,19],[78,73,7,46],[6,86,1,88]])
cost_org = copy.deepcopy(cost)
# Match using hungarian algorithm
print(len(cost))
for i in range(len(cost)):
    cost[i,:] = cost[i,:] - np.min(cost[i,:])
for i in range(len(cost)):
    cost[:,i] = cost[:,i] - np.min(cost[:,i])

mask = np.zeros([len(cost),len(cost)])
gou_mask = np.zeros([len(cost),len(cost)])

mask[cost > 0] = 1
gou_mask[cost > 0] = 1
mask = 1 - mask # zero elements are masked with 1
gou_mask = 1 - gou_mask
print(mask)

count = 1000000000
cou = np.count_nonzero(mask[i,:])
while cou !=count:
    # row
    count = cou
    for i in range(len(cost)):
        zero_num = np.count_nonzero(mask[i,:])
        if zero_num == 1:
            zero_index = np.where(mask[i,:] == 1)
            zero_col_index = np.where(mask[:,zero_index[0]] == 1)
            #print(zero_col_index)
            for col_index in zero_col_index[0]:
                #print(col_index)
                if col_index != i:
                    mask[i,col_index] = 0
            #print(zero_num,zero_index,zero_col_index)
    # col
    for i in range(len(cost)):
        zero_num = np.count_nonzero(mask[:,i])
        if zero_num == 1:
            zero_index = np.where(mask[:,i] == 1)
            zero_col_index = np.where(mask[zero_index[0],:] == 1)
            #print(zero_col_index)
            for col_index in zero_col_index[0]:
                #print(col_index)
                if col_index != i:
                    mask[col_index,i] = 0
            #print(zero_num,zero_index,zero_col_index)
    cou = np.count_nonzero(mask[i,:])
line_mask = np.zeros([len(cost),2])
# mask all rows which have no 1 in 'mask'
count = 0
for i in range(len(cost)):
    #print(np.count_nonzero(mask[i, :]))
    if np.count_nonzero(mask[i,:]) < 1:
        line_mask[i,0] = 1
#print(line_mask)
cou = np.count_nonzero(line_mask)
while cou != count:
    count = cou
    # mask all columns corresponding to the zero elements in the above rows
    for i in range(len(cost)):
        if line_mask[i,0] == 1:
            for j in range(len(cost)):
                if gou_mask[i,j] == 1:
                    line_mask[j,1] = 1

    # mask all rows corresponding to the zero elements in the above columns
    for i in range(len(cost)):
        if line_mask[i,1] == 1:
            for j in range(len(cost)):
                if mask[j,i] == 1:
                    line_mask[j,0] = 1
    cou = np.count_nonzero(line_mask)
    print(cou)
np.set_printoptions(threshold=40000)
row_line = np.count_nonzero(line_mask[:,0])
col_line = np.count_nonzero(line_mask[:,1])
line_all = len(cost)-row_line+col_line

print(row_line,col_line,line_all)
if line_all < len(cost):
    #find the smallest number in uncovered aera
    np.count_nonzero(line_mask[,:])
    pass

# optimal_val = find_optimal()
# cursively solve this problem


class Node(object):
    """节点类"""

    def __init__(self, elem_people=-1,elem_target=-1, child=None):
        self.elem_people = elem_people
        self.elem_target = elem_target
        self.child = child

def growTree(mask_this,i,n,target_mask):
    index_op = np.argwhere(mask_this[i,:] > 0)
    Nodes = []
    Tree = Node(i,n)
    for j in index_op:
        mask = target_mask
        if len(np.argwhere(target_mask == j))==0:
            if i + 1 < len(mask_this):
                mask.append(j)
                #print(mask)
                ans,ma = growTree(mask_this, i + 1, j, mask)
                Nodes.append(ans)
            else:
                mask.append(j)
                print(mask)
                ma = mask
                Nodes.append(Node(i+1,j))
    Tree.child = Nodes
    return Tree,ma

print(gou_mask)
target_mask = []
Tree,mask_route = growTree(gou_mask,0,-1,target_mask)
print(mask_route)

cost_optimal = 0
count = 0
for i in mask_route:
    cost_optimal = cost_optimal + cost_org[count, i[0]]
    print(cost_org[count][i[0]])
    count = count + 1

print(cost_optimal)



