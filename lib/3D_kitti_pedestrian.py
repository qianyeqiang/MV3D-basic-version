import os
import time
import PIL
import numpy as np
import scipy.sparse
import cv2
from sklearn.cluster import DBSCAN
from collections import Counter
#from utils.cython_bbox import bbox_overlaps
#from utils.boxes_grid import get_boxes_grid
#import subprocess
#import cPickle
#from fast_rcnn.config import cfg
#import math
#from rpn_msr.generate_anchors import generate_anchors_bv
from utils.transform import lidar_point_to_img, lidar_cnr_to_img, camera_to_lidar_cnr, lidar_to_corners_single,\
    computeCorners3D, lidar_3d_to_bv, lidar_cnr_to_3d,lidar_cnr_to_camera,corners_to_boxes
import mayavi.mlab as mlab

bool_ifshow = 1


def load_kitti_calib(index):
    """
    load projection matrix

    """
    data_path = '../data/KITTI/object/'
    prefix = 'training/calib'
    prefix = 'testing/calib'
    calib_dir = os.path.join(data_path, prefix, index + '.txt')

    with open(calib_dir) as fi:
        lines = fi.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

def load_kitti_annotation(index):
    """
    Load image and bounding boxes info from txt file in the KITTI
    format.
    """
    classes = ('__background__', 'Car', 'pedestrian', 'Cyclist')
    num_classes = len(classes)
    class_to_ind = dict(zip(classes, xrange(num_classes)))

    # filename = '$Faster-RCNN_TF/data/KITTI/object/training/label_2/000000.txt'
    data_path = '../data/KITTI/object/'
    filename = os.path.join(data_path, 'training/label_2', index + '.txt')

    data_path = 'mscnn_testing/'
    filename = os.path.join(data_path, index + '.txt')

    # data_path = 'result_jack/'
    # filename = os.path.join(data_path, index + '.txt')
#         print("Loading: ", filename)

    # calib
    calib = load_kitti_calib(index)
    Tr = calib['Tr_velo2cam']

    # print 'Loading: {}'.format(filename)
    with open(filename, 'r') as f:
        lines = f.readlines()
    num_objs = len(lines)
    translation = np.zeros((num_objs, 3), dtype=np.float32)
    rys = np.zeros((num_objs), dtype=np.float32)
    lwh = np.zeros((num_objs, 3), dtype=np.float32)
    boxes = np.zeros((num_objs, 4), dtype=np.float32)
    boxes_bv = np.zeros((num_objs, 4), dtype=np.float32)
    boxes3D = np.zeros((num_objs, 6), dtype=np.float32)
    boxes3D_lidar = np.zeros((num_objs, 6), dtype=np.float32)
    boxes3D_cam_cnr = np.zeros((num_objs, 24), dtype=np.float32)
    boxes3D_corners = np.zeros((num_objs, 24), dtype=np.float32)
    alphas = np.zeros((num_objs), dtype=np.float32)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, num_classes), dtype=np.float32)
    scores = np.zeros((num_objs), dtype=np.float32)

    # print(boxes3D.shape)

    # Load object bounding boxes into a data frame.

    ix = -1
    for line in lines:
        obj = line.strip().split(' ')
        try:

            cls = class_to_ind[obj[0].strip()]
        except:
            continue
        # ignore objects with undetermined difficult level
        # level = self._get_obj_level(obj)
        # if level > 3:
        #     continue
        ix += 1
        # 0-based coordinates
        alpha = float(obj[3])
        x1 = float(obj[4])
        y1 = float(obj[5])
        x2 = float(obj[6])
        y2 = float(obj[7])
        h = float(obj[8])
        w = float(obj[9])
        l = float(obj[10])
        tx = float(obj[11])
        ty = float(obj[12])
        tz = float(obj[13])
        ry = float(obj[14])
        score = float(obj[15])

        rys[ix] = ry
        lwh[ix, :] = [l, w, h]
        alphas[ix] = alpha
        translation[ix, :] = [tx, ty, tz]
        boxes[ix, :] = [x1, y1, x2, y2]
        boxes3D[ix, :] = [tx, ty, tz, l, w, h]
        # convert boxes3D cam to 8 corners(cam)
        boxes3D_cam_cnr_single = computeCorners3D(boxes3D[ix, :], ry)
        boxes3D_cam_cnr[ix, :] = boxes3D_cam_cnr_single.reshape(24)
        # convert 8 corners(cam) to 8 corners(lidar)
        boxes3D_corners[ix, :] = camera_to_lidar_cnr(boxes3D_cam_cnr_single, Tr)
        # convert 8 corners(cam) to  lidar boxes3D
        boxes3D_lidar[ix, :] = lidar_cnr_to_3d(boxes3D_corners[ix, :], lwh[ix,:])
        # convert 8 corners(lidar) to lidar bird view
        boxes_bv[ix, :] = lidar_3d_to_bv(boxes3D_lidar[ix, :])
        # boxes3D_corners[ix, :] = lidar_to_corners_single(boxes3D_lidar[ix, :])
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0
        scores[ix] = score

    rys.resize(ix+1)
    lwh.resize(ix+1, 3)
    translation.resize(ix+1, 3)
    alphas.resize(ix+1)
    boxes.resize(ix+1, 4)
    boxes_bv.resize(ix+1, 4)
    boxes3D.resize(ix+1, 6)
    boxes3D_lidar.resize(ix+1, 6)
    boxes3D_cam_cnr.resize(ix+1, 24)
    boxes3D_corners.resize(ix+1, 24)
    gt_classes.resize(ix+1)
    # print(self.num_classes)
    overlaps.resize(ix+1, num_classes)
    # if index == '000142':
    #     print(index)
    #     print(overlaps)
    overlaps = scipy.sparse.csr_matrix(overlaps)
    # if index == '000142':
    #     print(overlaps)
    scores.resize(ix+1)

    return {'ry' : rys,
            'lwh' : lwh,
            'boxes' : boxes,
            'boxes_bv' : boxes_bv,
            'boxes_3D_cam' : boxes3D,
            'boxes_3D' : boxes3D_lidar,
            'boxes3D_cam_corners' : boxes3D_cam_cnr,
            'boxes_corners' : boxes3D_corners,
            'gt_classes': gt_classes,
            'gt_overlaps' : overlaps,
            'xyz' : translation,
            'alphas' :alphas,
            'flipped' : False,
            'scores': scores}

def load_kittigt_annotation(index):
    """
    Load image and bounding boxes info from txt file in the KITTI
    format.
    """
    classes = ('__background__', 'Car', 'Pedestrian', 'Cyclist')
    num_classes = len(classes)
    class_to_ind = dict(zip(classes, xrange(num_classes)))

    # filename = '$Faster-RCNN_TF/data/KITTI/object/training/label_2/000000.txt'
    data_path = '../data/KITTI/object/'
    filename = os.path.join(data_path, 'training/label_2', index + '.txt')

    # calib
    calib = load_kitti_calib(index)
    Tr = calib['Tr_velo2cam']

    # print 'Loading: {}'.format(filename)
    with open(filename, 'r') as f:
        lines = f.readlines()
    num_objs = len(lines)
    translation = np.zeros((num_objs, 3), dtype=np.float32)
    rys = np.zeros((num_objs), dtype=np.float32)
    lwh = np.zeros((num_objs, 3), dtype=np.float32)
    boxes = np.zeros((num_objs, 4), dtype=np.float32)
    boxes_bv = np.zeros((num_objs, 4), dtype=np.float32)
    boxes3D = np.zeros((num_objs, 6), dtype=np.float32)
    boxes3D_lidar = np.zeros((num_objs, 6), dtype=np.float32)
    boxes3D_cam_cnr = np.zeros((num_objs, 24), dtype=np.float32)
    boxes3D_corners = np.zeros((num_objs, 24), dtype=np.float32)
    alphas = np.zeros((num_objs), dtype=np.float32)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, num_classes), dtype=np.float32)

    # print(boxes3D.shape)

    # Load object bounding boxes into a data frame.

    ix = -1
    for line in lines:
        obj = line.strip().split(' ')
        try:

            cls = class_to_ind[obj[0].strip()]
        except:
            continue
        # ignore objects with undetermined difficult level
        # level = self._get_obj_level(obj)
        # if level > 3:
        #     continue
        ix += 1
        # 0-based coordinates
        alpha = float(obj[3])
        x1 = float(obj[4])
        y1 = float(obj[5])
        x2 = float(obj[6])
        y2 = float(obj[7])
        h = float(obj[8])
        w = float(obj[9])
        l = float(obj[10])
        tx = float(obj[11])
        ty = float(obj[12])
        tz = float(obj[13])
        ry = float(obj[14])

        rys[ix] = ry
        lwh[ix, :] = [l, w, h]
        alphas[ix] = alpha
        translation[ix, :] = [tx, ty, tz]
        boxes[ix, :] = [x1, y1, x2, y2]
        boxes3D[ix, :] = [tx, ty, tz, l, w, h]
        # convert boxes3D cam to 8 corners(cam)
        boxes3D_cam_cnr_single = computeCorners3D(boxes3D[ix, :], ry)
        boxes3D_cam_cnr[ix, :] = boxes3D_cam_cnr_single.reshape(24)
        # convert 8 corners(cam) to 8 corners(lidar)
        boxes3D_corners[ix, :] = camera_to_lidar_cnr(boxes3D_cam_cnr_single, Tr)
        # convert 8 corners(cam) to  lidar boxes3D
        boxes3D_lidar[ix, :] = lidar_cnr_to_3d(boxes3D_corners[ix, :], lwh[ix,:])
        # convert 8 corners(lidar) to lidar bird view
        boxes_bv[ix, :] = lidar_3d_to_bv(boxes3D_lidar[ix, :])
        # boxes3D_corners[ix, :] = lidar_to_corners_single(boxes3D_lidar[ix, :])
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0

    rys.resize(ix+1)
    lwh.resize(ix+1, 3)
    translation.resize(ix+1, 3)
    alphas.resize(ix+1)
    boxes.resize(ix+1, 4)
    boxes_bv.resize(ix+1, 4)
    boxes3D.resize(ix+1, 6)
    boxes3D_lidar.resize(ix+1, 6)
    boxes3D_cam_cnr.resize(ix+1, 24)
    boxes3D_corners.resize(ix+1, 24)
    gt_classes.resize(ix+1)
    # print(self.num_classes)
    overlaps.resize(ix+1, num_classes)
    # if index == '000142':
    #     print(index)
    #     print(overlaps)
    overlaps = scipy.sparse.csr_matrix(overlaps)
    # if index == '000142':
    #     print(overlaps)

    return {'ry' : rys,
            'lwh' : lwh,
            'boxes' : boxes,
            'boxes_bv' : boxes_bv,
            'boxes_3D_cam' : boxes3D,
            'boxes_3D' : boxes3D_lidar,
            'boxes3D_cam_corners' : boxes3D_cam_cnr,
            'boxes_corners' : boxes3D_corners,
            'gt_classes': gt_classes,
            'gt_overlaps' : overlaps,
            'xyz' : translation,
            'alphas' :alphas,
            'flipped' : False}

def load_image_set_index():
    """
    Load the indexes listed in this dataset's image set file.
    """
    image_set_file = '../data/KITTI/ImageSets/imageset_testing7518.txt'
    #image_set_file = '../start.txt'
    #image_set_file = os.path.join(kitti_path, 'ImageSets',self._image_set + '.txt')
    assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)

    with open(image_set_file) as f:
        image_index = [x.rstrip('\n') for x in f.readlines()]

    print 'image sets length: ', len(image_index)
    return image_index

def lidar3D_path_from_index(index):
    """
    Construct an image path from the image's "index" identifier.
    """
    # set the prefix
    prefix = 'training/velodyne'
    #prefix = 'testing/velodyne'
    data_path = '../data/KITTI/object/'
    # lidar_bv_path = '$Faster-RCNN_TF/data/KITTI/object/training/lidar_bv/000000.npy'
    lidar3D_path = os.path.join(data_path, prefix, index + '.bin')
    assert os.path.exists(lidar3D_path), \
            'Path does not exist: {}'.format(lidar3D_path)
    return lidar3D_path

def image_path_from_index(index):
    """
    Construct an image path from the image's "index" identifier.
    """
    # set the prefix
    prefix = 'training/image_2/'
    prefix = 'testing/image_2/'
    data_path = '../data/KITTI/object/'
    # lidar_bv_path = '$Faster-RCNN_TF/data/KITTI/object/training/lidar_bv/000000.npy'
    image_path = os.path.join(data_path, prefix, index + '.png')
    assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
    return image_path

def draw_lidar(lidar, is_grid=False, is_top_region=False, fig=None):
    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    if fig is None: fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
    mlab.points3d(
        #pxs, pys, pzs, prs,
        pxs, pys, pzs,
        mode='point',  # 'point'  'sphere'
        colormap='spectral',  #'gnuplot', 'bone',  #'spectral',  #'copper',
        scale_factor=10,
        figure=fig)

    mlab.orientation_axes()
    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991

    if 1:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        axes=np.array([
            [2.,0.,0.,0.],
            [0.,2.,0.,0.],
            [0.,0.,2.,0.],
        ],dtype=np.float64)
        fov=np.array([  ##<todo> : now is 45 deg. use actual setting later ...
            [20., 20., 0.,0.],
            [20.,-20., 0.,0.],
        ],dtype=np.float64)


        mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
        mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)


def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,1,1), line_width=0.1):

    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]

        #mlab.text3d(b[0,0], b[0,1], b[0,2], '%d'%n, scale=(1, 1, 1), color=color, figure=fig)
        for k in range(0,4):

            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991

def dbscan(point_can, point):

    db = DBSCAN(eps=0.13, min_samples=1).fit(point_can)
    labels = db.labels_
    dict_labels = Counter(labels)
    max_label_num = np.max(dict_labels.values())

    for key, value in dict_labels.iteritems():
        if value == max_label_num:
            key_label = key

    index_labels = labels == key_label
    point_can = point_can[index_labels]

    x = 0
    y = 0
    z = 0
    low_z = 0
    point_num = point_can.shape[0]

    for i in range(point_num):
        x = x + point_can[i, 0]
        y = y + point_can[i, 1]
        z = z + point_can[i, 2]
    x = x/point_num
    y = y/point_num
    z = z/point_num

    #print x,y,z
    index = (point[:, 0] > x-0.6) * (point[:, 0] < x+0.6) * (point[:, 1] > y-0.6) * (point[:, 1] < y+0.6)\
        * (point[:, 2] > z-1.5) * (point[:, 2] < z+2)
    point_pede = point[index,:]

    for k in range(point_pede.shape[0]):
        if point_pede[k, 2]< low_z:
            low_z = point_pede[k, 2]

    index = point_pede[:, 2]>low_z+0.1
    point_pede = point_pede[index,:]

    if len(point_pede)==0:
        return point_pede
    db2 = DBSCAN(eps=0.3, min_samples=1).fit(point_pede)
    db2_labels = db2.labels_
    db2_dict_labels = Counter(db2_labels)

    db2_max_label_num = np.max(db2_dict_labels.values())

    for key, value in db2_dict_labels.iteritems():
        if value == db2_max_label_num:
            db2_key_label = key

    db2_index_labels = db2_labels == db2_key_label
    point_pede_single = point_pede[db2_index_labels]

    return point_pede_single

def calculate_corner(point):

    x_min = 100
    x_max = 0
    y_min = 100
    y_max = -100
    z_min = 100
    z_max = -100

    for i in range(point.shape[0]):
        if point[i, 0]<x_min:
            x_min = point[i, 0]
        if point[i, 0]>x_max:
            x_max = point[i, 0]
        if point[i, 1]<y_min:
            y_min = point[i, 1]
        if point[i, 1]>y_max:
            y_max = point[i, 1]
        if point[i, 2]<z_min:
            z_min = point[i, 2]
        if point[i, 2]>z_max:
            z_max = point[i, 2]

    corners = np.zeros(((1,8,3)))

    z_max = z_max+0.1
    # if z_min>-1.6:
    #     z_min = -1.6

    corners[0, 0, 0] = x_max
    corners[0, 0, 1] = y_min
    corners[0, 0, 2] = z_max
    corners[0, 1, 0] = x_min
    corners[0, 1, 1] = y_min
    corners[0, 1, 2] = z_max
    corners[0, 2, 0] = x_min
    corners[0, 2, 1] = y_max
    corners[0, 2, 2] = z_max
    corners[0, 3, 0] = x_max
    corners[0, 3, 1] = y_max
    corners[0, 3, 2] = z_max
    corners[0, 4, 0] = x_max
    corners[0, 4, 1] = y_min
    corners[0, 4, 2] = z_min
    corners[0, 5, 0] = x_min
    corners[0, 5, 1] = y_min
    corners[0, 5, 2] = z_min
    corners[0, 6, 0] = x_min
    corners[0, 6, 1] = y_max
    corners[0, 6, 2] = z_min
    corners[0, 7, 0] = x_max
    corners[0, 7, 1] = y_max
    corners[0, 7, 2] = z_min

    #print corners.shape
    return corners

def calib_at(index):
    """
    Return the calib sequence.
    """
    calib_ori = load_kitti_calib(index)
    calib = np.zeros((4, 12))
    calib[0,:] = calib_ori['P2'].reshape(12)
    calib[1,:] = calib_ori['P3'].reshape(12)
    calib[2,:9] = calib_ori['R0'].reshape(9)
    calib[3,:] = calib_ori['Tr_velo2cam'].reshape(12)

    return calib

def write_result(image_index,box_img,camera_box3D,scores):
    type = "pedestrian"
    truncation = -1
    occlusion = -1
    alpha = -10

    x1 = box_img[0]
    y1 = box_img[1]
    x2 = box_img[2]
    y2 = box_img[3]

    h = camera_box3D[0]
    w = camera_box3D[1]
    l = camera_box3D[2]
    x = camera_box3D[3]
    y = camera_box3D[4]
    z = camera_box3D[5]
    ry = camera_box3D[6]

    score = scores

    write = open("result_jack/" + image_index + ".txt", "a")


    write.write(type+" "+str(truncation)+" "+str(occlusion)+" "+str(alpha)+" "+\
                str(x1)+" "+str(y1)+" "+str(x2)+" "+str(y2)+" "+\
                str(h)+" "+str(w)+" "+str(l)+" "+\
                str(x)+" "+str(y)+" "+str(z)+" "+str(ry)+" "+str(score)+'\n')

####### create the whole empty file
# for i in range(7518):
#     image_index = load_image_set_index()
#     image_count = image_index[i]
#     write = open("result_jack/" + image_count + ".txt", "a")

for i in range(10):

    image_index = load_image_set_index()
    print image_index[i]
    image_count = image_index[i]

    ######### draw lidar figure
    if bool_ifshow == 1:
        fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
        lidar3D = lidar3D_path_from_index(image_index[i])
        filename = lidar3D
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        draw_lidar(scan, fig=fig)


    ######## draw lidar corners_gt
    if bool_ifshow == 1:
        GT_boxes3D_corners = load_kittigt_annotation(image_index[i])["boxes_corners"]
        GT_corners = GT_boxes3D_corners[:, :24].reshape((-1, 3, 8)).transpose((0, 2, 1))
        draw_gt_boxes3d(GT_corners, color=(1, 0, 1), fig=fig)
        #mlab.show()
        pass

    ####### draw image bbox

    # calib = calib_at(image_index[i])
    # img_boxes = lidar_cnr_to_img(GT_boxes3D_corners[:, :24], calib[3], calib[2], calib[0])
    # img_boexs_gt = load_kitti_annotation(image_index[i])["boxes"]
    # image_name = image_path_from_index(image_index[i])
    # image = cv2.imread(image_name)
    # for i in range(img_boxes.shape[0]):
    #     if gt_classes[i] != 2:
    #         continue
    #     #cv2.rectangle(image, (img_boxes[i,0], img_boxes[i,1]), (img_boxes[i,2], img_boxes[i,3]), (0, 255, 0))
    #     cv2.rectangle(image, (img_boexs_gt[i,0], img_boexs_gt[i,1]), (img_boexs_gt[i,2], img_boexs_gt[i,3]), (255, 255, 255))
    #     pass

    #cv2.imshow("image", image)
    #cv2.waitKey()


    ########### draw points

    calib = calib_at(image_index[i])

    # image_name = image_path_from_index(image_index[i])
    # image = cv2.imread(image_name)
    # point  = np.array([[22.78300095,  -6.82499981,  -1.26900005]])
    # point_img = lidar_point_to_img(point , calib[3], calib[2], calib[0])
    # cv2.circle(image, (int(point_img[0]), int(point_img[1])), 2, (0, 0, 255), -1)

    image_name = image_path_from_index(image_index[i])
    image = cv2.imread(image_name)

    box_img = load_kitti_annotation(image_index[i])["boxes"]
    scores = load_kitti_annotation(image_index[i])["scores"]
    #print "box_img.shape: ", box_img.shape

    lidar3D_filename = lidar3D_path_from_index(image_index[i])
    scan = np.fromfile(lidar3D_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    #print "scan.shape: ", scan.shape
    point = scan[:, :3]
    point_img = lidar_point_to_img(point, calib[3], calib[2], calib[0])
    point_img = point_img.transpose((1, 0))
    #print "point_img.shape: ", point_img.shape

    gt_classes = load_kitti_annotation(image_index[i])["gt_classes"]
    for i in range(box_img.shape[0]):
        if gt_classes[i] != 2:
            continue
        x_resu = (box_img[i,2] - box_img[i, 0])/100
        y_resu = (box_img[i,3] - box_img[i, 1])/100
        xmin = box_img[i, 0]+x_resu*45
        xmax = box_img[i, 2]-x_resu*45
        ymin = box_img[i, 1]+y_resu*30
        ymax = box_img[i, 3]-y_resu*50

        index = (point_img[:, 0]>xmin) * (point_img[:, 0]<xmax) *\
             (point_img[:, 1]>ymin) * (point_img[:, 1]<ymax) * (point[:, 0]>0)

        point_img_can = point_img[index, :]
        point_can = point[index, :]
        if bool_ifshow == 1:
            for k in range(point_img_can.shape[0]):
                cv2.circle(image, (int(point_img_can[k, 0]), int(point_img_can[k, 1])), 2, (0, 0, 255), -1)

            cv2.rectangle(image, (int(box_img[i,2]), int(box_img[i,3])), (int(box_img[i, 0]), int(box_img[i, 1])),(255, 255, 255))
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 255), thickness=1)


        if len(point_can)>0:
            point_pede = dbscan(point_can, point)
            #point_pede = point_can
            if len(point_pede) != 0:
                if bool_ifshow == 1:
                    mlab.points3d(
                        # pxs, pys, pzs, prs,
                        point_pede[:, 0], point_pede[:, 1], point_pede[:, 2],
                        mode='point',  # 'point'  'sphere'
                        #colormap='spectral',  # 'gnuplot', 'bone',  #'spectral',  #'copper',
                        #scale_factor=10,
                        color = (1, 0, 0),
                        figure=fig)

                detect_corner = calculate_corner(point_pede)
                if bool_ifshow == 1:
                    draw_gt_boxes3d(detect_corner, color=(0, 1, 0), fig=fig)

                detect_corner = detect_corner.transpose((0,2,1))

                camera_box3D = corners_to_boxes(lidar_cnr_to_camera(detect_corner, calib[3]))
                if scores[i]>0.0:
                    write_result(image_count,box_img[i],camera_box3D,scores[i])
                    pass

    if bool_ifshow == 1:
        cv2.imshow("image", image)
        mlab.show()
        cv2.waitKey()



