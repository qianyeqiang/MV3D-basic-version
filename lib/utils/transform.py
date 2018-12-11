import numpy as np

TOP_X_MAX = 60
TOP_X_MIN = 0
TOP_Y_MIN = -30
TOP_Y_MAX = 30
RES = 0.1
LIDAR_HEIGHT = 1.73
CAR_HEIGHT = 1.56
X0, Xn = 0, int((TOP_X_MAX - TOP_X_MIN) // RES) + 1
Y0, Yn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // RES) + 1

def _lidar_to_bv_coord(x, y):
    X0, Xn = 0, int((TOP_X_MAX - TOP_X_MIN) // RES) + 1
    Y0, Yn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // RES) + 1

    xx = Yn - (y - TOP_Y_MIN) // RES
    yy = Xn - (x - TOP_X_MIN) // RES

    return xx, yy

def lidar_cnr_to_bv_single(corners):
    pts_2D = np.zeros(4)
    # min & max in lidar coords
    xmin = np.min(corners[:8])
    xmax = np.max(corners[:8])
    ymin = np.min(corners[8:16])
    ymax = np.max(corners[8:16])

    # top left bottom right at lidar bird view coords
    pts_2D = np.array([xmax, ymax, xmin, ymin])

    pts_2D[0], pts_2D[1] = _lidar_to_bv_coord(pts_2D[0], pts_2D[1])
    pts_2D[2], pts_2D[3] = _lidar_to_bv_coord(pts_2D[2], pts_2D[3])

    return pts_2D

#  def lidar_cnr_to_bv_single(corners):
    #  """
    #  convert lidar corners (x0-x7,y0-y7,z0-z7) to lidar bv view (x1 ,y1, x2, y2)
    #  """
    #  if corners.shape == (3, 8):
        #  corners = corners.reshape(24)

    #  assert corners.shape[0] == 24
    #  pts_2D = np.zeros(4)
    #  xmin = np.min(corners[:8])
    #  xmax = np.max(corners[:8])
    #  ymin = np.min(corners[8:16])
    #  ymax = np.max(corners[8:16])

    #  pts_2D = np.array([xmin, ymin, xmax, ymax])

    #  pts_2D[0], pts_2D[1] = _lidar_to_bv_coord(pts_2D[0], pts_2D[1])
    #  pts_2D[2], pts_2D[3] = _lidar_to_bv_coord(pts_2D[2], pts_2D[3])

    #  return pts_2D

def lidar_to_bv_single(rois_3d):
    """
    cast lidar 3d points(x, y, z, l, w, h) to bird view (x1, y1, x2, y2)
    """
    assert(rois_3d.shape[0] == 6)
    rois = np.zeros((4))

    rois[0] = rois_3d[0] + rois_3d[3] * 0.5
    rois[1] = rois_3d[1] + rois_3d[4] * 0.5
    rois[2] = rois_3d[0] - rois_3d[3] * 0.5
    rois[3] = rois_3d[1] - rois_3d[4] * 0.5

    rois[0], rois[1] = _lidar_to_bv_coord(rois[0], rois[1])
    rois[2], rois[3] = _lidar_to_bv_coord(rois[2], rois[3])
    # print rois
    # assert rois[0] < 1000
    # assert rois[2] < 1000
    # assert rois[1] < 1000
    # assert rois[3] < 1000
    return rois


def _bv_to_lidar_coords(xx,yy):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//RES)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//RES)+1
    y = Xn*RES-(xx+0.5)*RES + TOP_Y_MIN
    x = Yn*RES-(yy+0.5)*RES + TOP_X_MIN

    return x,y

def bv_anchor_to_lidar(anchors):
    """
    convert 2d anchors to 3d anchors
    """
    ex_lengths = anchors[:, 3] - anchors[:, 1]
    ex_widths = anchors[:, 2] - anchors[:, 0]

    ex_ctr_x = (anchors[:,0] + anchors[:,2]) / 2.
    ex_ctr_y = (anchors[:,1] + anchors[:,3]) / 2.

    ex_lengths = ex_lengths.reshape((anchors.shape[0], 1)) * RES
    ex_widths = ex_widths.reshape((anchors.shape[0], 1)) * RES
    ex_ctr_x = ex_ctr_x.reshape((anchors.shape[0], 1))
    ex_ctr_y = ex_ctr_y.reshape((anchors.shape[0], 1))

    ex_ctr_x, ex_ctr_y = _bv_to_lidar_coords(ex_ctr_x, ex_ctr_y)

    ex_heights = np.ones((anchors.shape[0], 1), dtype=np.float32) * CAR_HEIGHT
    ex_ctr_z = np.ones((anchors.shape[0], 1), dtype=np.float32) * -(LIDAR_HEIGHT-CAR_HEIGHT/2.) #

    anchors_3d = np.hstack((ex_ctr_x, ex_ctr_y, ex_ctr_z, ex_lengths, ex_widths, ex_heights))

    return anchors_3d

def lidar_3d_to_bv(rois_3d):
    """
    cast lidar 3d points(x, y, z, l, w, h) to bird view (x1, y1, x2, y2)
    """

    if len(rois_3d.shape) == 1:
    # if 0:
        rois = np.zeros(4)

        rois[0] = rois_3d[0] + rois_3d[3] * 0.5
        rois[1] = rois_3d[1] + rois_3d[4] * 0.5
        rois[2] = rois_3d[0] - rois_3d[3] * 0.5
        rois[3] = rois_3d[1] - rois_3d[4] * 0.5

        rois[0], rois[1] = _lidar_to_bv_coord(rois[0], rois[1])
        rois[2], rois[3] = _lidar_to_bv_coord(rois[2], rois[3])

    else:

        rois = np.zeros((rois_3d.shape[0], 4))

        rois[:, 0] = rois_3d[:, 0] + rois_3d[:, 3] * 0.5
        rois[:, 1] = rois_3d[:, 1] + rois_3d[:, 4] * 0.5
        rois[:, 2] = rois_3d[:, 0] - rois_3d[:, 3] * 0.5
        rois[:, 3] = rois_3d[:, 1] - rois_3d[:, 4] * 0.5

        rois[:, 0], rois[:, 1] = _lidar_to_bv_coord(rois[:, 0], rois[:, 1])
        rois[:, 2], rois[:, 3] = _lidar_to_bv_coord(rois[:, 2], rois[:, 3])

    return rois.astype(np.float32)


def lidar_to_bv(rois_3d):
    """
    cast lidar 3d points(0,x, y, z, l, w, h) to bird view (0,x1, y1, x2, y2)
    """

    rois = np.zeros((rois_3d.shape[0], 5))
    rois[:, 0] = rois_3d[:, 0]

    rois[:, 1] = rois_3d[:, 1] + rois_3d[:, 4] * 0.5
    rois[:, 2] = rois_3d[:, 2] + rois_3d[:, 5] * 0.5
    rois[:, 3] = rois_3d[:, 1] - rois_3d[:, 4] * 0.5
    rois[:, 4] = rois_3d[:, 2] - rois_3d[:, 5] * 0.5

    rois[:, 1], rois[:, 2] = _lidar_to_bv_coord(rois[:, 1], rois[:, 2])
    rois[:, 3], rois[:, 4] = _lidar_to_bv_coord(rois[:, 3], rois[:, 4])

    return rois.astype(np.float32)


def _bv_to_lidar_coord(x, y):
    Y0, Yn = 0, int((TOP_X_MAX - TOP_X_MIN) // RES) + 1
    X0, Xn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // RES) + 1
    yy = (Yn - y) * RES + TOP_Y_MIN
    xx = (Xn - x) * RES + TOP_X_MIN
    return xx, yy


def lidar_cnr_to_3d(corners, lwh):
    """
    lidar_corners to Boxex3D
    """
    shape = corners.shape
    if shape[0] == 24:
        boxes_3d = np.zeros(6)
        corners = corners.reshape((3, 8))
        boxes_3d[:3] = corners.mean(1)
        boxes_3d[3:] = lwh
    else:
        boxes_3d = np.zeros((shape[0],6))
        corners = corners.reshape((-1, 3, 8))
        boxes_3d[:,:3] = corners.mean(2)
        boxes_3d[:,3:] = lwh
    return boxes_3d

def cam_to_lidar_3d(pts_3D, Tr):
    """
    convert camera(x, y, z, l, w, h) to lidar (x, y, z, l, w, h)
    """

    points = pts_3D[:,:3].transpose()
    points = np.vstack((points, np.zeros(pts_3D.shape[0])))

    R = np.linalg.inv(Tr[:, :3])
    T = np.zeros((3, 1))
    T[0] = -Tr[1,3]
    T[1] = -Tr[2,3]
    T[2] = Tr[0,3]
    RT = np.hstack((R, T))
    points_lidar = np.dot(RT, points)

    pts_3D_lidar = np.zeros((pts_3D.shape))
    pts_3D_lidar[:,:3] = points_lidar.transpose()
    pts_3D_lidar[:,3:6] = pts_3D[:,3:6]

    return pts_3D_lidar

def cam_to_lidar_3d_single(pts_3D,Tr):
    """
    convert camera(x, y, z, l, w, h) to lidar (x, y, z, l, w, h)
    """

    points = pts_3D[:3]
    points = np.vstack((points, 0)).reshape((4, 1))

    R = np.linalg.inv(Tr[:, :3])
    T = np.zeros((3, 1))
    T[0] = -Tr[1,3]
    T[1] = -Tr[2,3]
    T[2] = Tr[0,3]
    RT = np.hstack((R, T))

    points_lidar = np.dot(RT, points)

    pts_3D_lidar = np.zeros(6)
    pts_3D_lidar[:3] = points_lidar.flatten()
    pts_3D_lidar[3:6] = pts_3D[3:6]

    return pts_3D_lidar

def _bv_roi_to_3d(rpn_rois_bv):
    """ convert rpn rois (0, x1, y1, x2, y2) to lidar 3d anchor (0, x, y, z, l, w, h) """
     # convert bird view rpn_rois to lidar coordinates
    rpn_rois_bv[:,1], rpn_rois_bv[:,2] = _bv_to_lidar(rpn_rois_bv[:,1], rpn_rois_bv[:,2])
    rpn_rois_bv[:,3], rpn_rois_bv[:,4] = _bv_to_lidar(rpn_rois_bv[:,3], rpn_rois_bv[:,4])

    # convert rpn_rois(0, x1, y1, x2, y2) to rpn_rois_ctr (0, x, y, l, w)
    rpn_rois_ctr = np.zeros(shape=rpn_rois_bv.shape, dtype=rpn_rois_bv.dtype)

    rpn_rois_ctr[:,1] = (rpn_rois_bv[:,1] + rpn_rois_bv[:,3]) * 0.5 # x
    rpn_rois_ctr[:,2] = (rpn_rois_bv[:,2] + rpn_rois_bv[:,4]) * 0.5 # y
    rpn_rois_ctr[:,3] = np.abs(rpn_rois_bv[:,3] - rpn_rois_bv[:,1]) # l
    rpn_rois_ctr[:,4] = np.abs(rpn_rois_bv[:,2] - rpn_rois_bv[:,4]) # w

    # extend rpn_rois_ctr (0, x, y) to 3d rois (0, x, y, z, l, w, h)
    rshape = rpn_rois_ctr.shape
    ctr_height = np.ones((rshape[0]))*(- (LIDAR_HEIGHT - CAR_HEIGHT * 0.5))
    car_height = np.ones((rshape[0]))*CAR_HEIGHT
    ctr_height = ctr_height.reshape(-1, 1)
    car_height = car_height.reshape(-1, 1)

    if DEBUG:
        print rpn_rois_ctr.shape
        print 'car height shape', car_height.shape
        print 'ctr shape', ctr_height.shape

    all_rois_3d = np.hstack((rpn_rois_ctr[:,:3],
                ctr_height, rpn_rois_ctr[:,3:5], car_height))
    assert(all_rois_3d.shape[1] == 7)
    return all_rois_3d


def lidar_to_corners_single(pts_3D):
    """
    convert pts_3D_lidar (x, y, z, l, w, h) to
    8 corners (x0, ... x7, y0, ...y7, z0, ... z7)

    (x0, y0, z0) at left,forward, up.
    clock-wise
    """
    l = pts_3D[3]
    w = pts_3D[4]
    h = pts_3D[5]

    x_corners = np.array([l/2., l/2., -l/2., -l/2., l/2., l/2., -l/2., -l/2.])
    y_corners = np.array([w/2., -w/2., -w/2., w/2., w/2., -w/2., -w/2., w/2.])
    z_corners = np.array([h/2., h/2.,h/2.,h/2.,h/2.,h/2.,h/2.,h/2.])

    corners = np.vstack((x_corners, y_corners, z_corners))

    corners[0,:] = corners[0,:] + pts_3D[0]
    corners[1,:] = corners[1,:] + pts_3D[1]
    corners[2,:] = corners[2,:] + pts_3D[2]

    return corners.reshape(-1).astype(np.float32)

def lidar_3d_to_corners(pts_3D):
    """
    convert pts_3D_lidar (x, y, z, l, w, h) to
    8 corners (x0, ... x7, y0, ...y7, z0, ... z7)
    """

    l = pts_3D[:, 3]
    w = pts_3D[:, 4]
    h = pts_3D[:, 5]

    l = l.reshape(-1, 1)
    w = w.reshape(-1, 1)
    h = h.reshape(-1, 1)

    # clockwise, zero at bottom left
    x_corners = np.hstack((l/2., l/2., -l/2., -l/2., l/2., l/2., -l/2., -l/2.))
    y_corners = np.hstack((w/2., -w/2., -w/2., w/2., w/2., -w/2., -w/2., w/2.))
    z_corners = np.hstack((-h/2.,-h/2.,-h/2.,-h/2.,h/2.,h/2.,h/2.,h/2.))

    corners = np.hstack((x_corners, y_corners, z_corners))

    corners[:,0:8] = corners[:,0:8] + pts_3D[:,0].reshape((-1, 1)).repeat(8, axis=1)
    corners[:,8:16] = corners[:,8:16] + pts_3D[:,1].reshape((-1, 1)).repeat(8, axis=1)
    corners[:,16:24] = corners[:,16:24] + pts_3D[:,2].reshape((-1, 1)).repeat(8, axis=1)

    return corners

def projectToImage(pts_3D, P):
    """
    PROJECTTOIMAGE projects 3D points in given coordinate system in the image
    plane using the given projection matrix P.

    Usage: pts_2D = projectToImage(pts_3D, P)
    input: pts_3D: 3xn matrix
          P:      3x4 projection matrix
    output: pts_2D: 2xn matrix

    last edited on: 2012-02-27
    Philip Lenz - lenz@kit.edu
    """
    # project in image
    mat = np.vstack((pts_3D, np.ones((pts_3D.shape[1]))))

    pts_2D = np.dot(P, mat)

    # scale projected points
    pts_2D[0, :] = pts_2D[0, :] / pts_2D[2, :]
    pts_2D[1, :] = pts_2D[1, :] / pts_2D[2, :]
    pts_2D = np.delete(pts_2D, 2, 0)

    return pts_2D

def _corners_to_bv(corners):
    pts_2D = np.zeros((corners.shape[0], 4))

    # min & max in lidar coords
    xmin = np.min(corners[:, :8], axis=1).reshape(-1, 1)
    xmax = np.max(corners[:, :8], axis=1).reshape(-1, 1)
    ymin = np.min(corners[:, 8:16], axis=1).reshape(-1, 1)
    ymax = np.max(corners[:, 8:16], axis=1).reshape(-1, 1)

    # top left bottom right at lidar bird view coords
    pts_2D = np.hstack([xmax, ymax, xmin, ymin])

    pts_2D[:,0], pts_2D[:,1] = _lidar_to_bv_coord(pts_2D[:,0], pts_2D[:,1])
    pts_2D[:,2], pts_2D[:,3] = _lidar_to_bv_coord(pts_2D[:,2], pts_2D[:,3])

    return pts_2D

def corners_to_bv(corners):
    num_class = corners.shape[1] / 24
    bv = np.zeros((corners.shape[0], 4*num_class))
    for i in xrange(num_class):
        cnr = corners[:, i*24:(i+1)*24]
        bv[:, i*4:(i+1)*4] = _corners_to_bv(cnr)

    return bv


# def lidar_cnr_to_img(corners, Tr, R0, P2):
#     #lidar8-image4
#
#     img_boxes = np.zeros((corners.shape[0], 4))
#
#     P2 = P2.reshape((3, 4))
#     R0 = R0.reshape((4, 3))
#     Tr = Tr.reshape((3, 4))
#
#
#     new_corners = corners.reshape((-1, 3, 8))
#
#     mat = reduce(np.dot, [P2, R0, Tr])
#
#     #  img_cor = np.array(map(lambda x:np.dot(mat, np.vstack((x,np.zeros(8)))), new_corners))
#     new_corners = np.append(new_corners, np.zeros((new_corners.shape[0], 1, 8)), 1)
#     img_cor = np.dot(mat, new_corners).transpose(1, 0, 2)
#
#     xs = img_cor[:,0,:] / np.abs(img_cor[:,2,:])
#     ys = img_cor[:,1,:] / np.abs(img_cor[:,2,:])
#
#     xmin = np.min(xs, axis=1).reshape(-1, 1)
#     xmax = np.max(xs, axis=1).reshape(-1, 1)
#     ymin = np.min(ys, axis=1).reshape(-1, 1)
#     ymax = np.max(ys, axis=1).reshape(-1, 1)
#
#     img_boxes = np.hstack((xmin, ymin, xmax, ymax))
#
#     return img_boxes#.astype(np.int32)

#  def lidar_cnr_to_img(corners, Tr, R0, P2):

    #  img_boxes = np.zeros((corners.shape[0], 4))

    #  Tr = Tr.reshape((3, 4))
    #  R0 = R0.reshape((4, 3))
    #  P2 = P2.reshape((3, 4))

    #  for i in range(corners.shape[0]):

        #  img_cor = lidar_cnr_to_img_single(corners[i], Tr, R0, P2)

        #  img_cor = img_cor / np.abs(img_cor[2])

        #  #  xmin = np.max((0, np.min(img_cor[0])))
        #  xmin = np.min(img_cor[0])
        #  xmax = np.max(img_cor[0])
        #  #  ymin = np.max((0, np.min(img_cor[1])))
        #  ymin = np.min(img_cor[1])
        #  ymax = np.max(img_cor[1])

        #  img_boxes[i, :] = np.array([xmin, ymin, xmax, ymax])

    #  return img_boxes.astype(np.int32)

def computeCorners3D(Boxex3D, ry):

    # compute rotational matrix around yaw axis
    R = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1,               0],
         [-np.sin(ry), 0, np.cos(ry)]]).reshape((3,3))

    # 3D bounding box dimensions
    l, w, h = Boxex3D[3:6]
    x, y, z = Boxex3D[0:3]

    # 3D bounding box corners
    x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
    y_corners = np.array([0,0,0,0,-h,-h,-h,-h])
    z_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])

    corners = np.vstack((x_corners, y_corners, z_corners))

    # rotate and translate 3D bounding box
    corners_3D = np.dot(R, corners)
    corners_3D[0,:] = corners_3D[0,:] + x
    corners_3D[1,:] = corners_3D[1,:] + y
    corners_3D[2,:] = corners_3D[2,:] + z

    return corners_3D

def corners_to_bv_single(corners):
    pts_2D = np.zeros(4)
    # min & max in lidar coords
    xmin = np.min(corners[:8])
    xmax = np.max(corners[:8])
    ymin = np.min(corners[8:16])
    ymax = np.max(corners[8:16])

    # top left bottom right at lidar bird view coords
    pts_2D = np.array([xmax, ymax, xmin, ymin])

    pts_2D[0], pts_2D[1] = _lidar_to_bv_coord(pts_2D[0], pts_2D[1])
    pts_2D[2], pts_2D[3] = _lidar_to_bv_coord(pts_2D[2], pts_2D[3])

    return pts_2D

def lidar_cnr_to_img_single(corners, Tr, R0, P2):

    #rewrite by jackqian

    P2 = P2.reshape((3, 4))
    R0 = R0.reshape((4, 3))
    Tr = Tr.reshape((3, 4))

    T = np.zeros((1,4))
    T[0,3] = 1

    P2 = np.vstack((P2, T))
    Tr = np.vstack((Tr, T))

    T2 = np.zeros((4,1))
    T2[3,0] = 1
    R0 = np.hstack((R0, T2))

    assert Tr.shape == (4, 4)
    assert R0.shape == (4, 4)
    assert P2.shape == (4, 4)

    if 24 in corners.shape:
        corners = corners.reshape((3, 8))

    corners = np.vstack((corners, np.ones(8)))

    mat1 =  np.dot(P2, R0)
    mat2 = np.dot(mat1, Tr)
    img_cor = np.dot(mat2, corners)

    return img_cor

def lidar_cnr_to_img(corners, Tr, R0, P2):

    img_boxes = np.zeros((corners.shape[0], 4))

    for i in range(corners.shape[0]):

        img_cor = lidar_cnr_to_img_single(corners[i], Tr, R0, P2)

        img_cor = img_cor / img_cor[2]

        xmin = np.min(img_cor[0])
        xmax = np.max(img_cor[0])
        ymin = np.min(img_cor[1])
        ymax = np.max(img_cor[1])

        img_boxes[i, :] = np.array([xmin, ymin, xmax, ymax])

    return img_boxes.astype(np.int32)

def lidar_point_to_img(point, Tr, R0, P2):

    #rewrite by jackqian

    P2 = P2.reshape((3, 4))
    R0 = R0.reshape((4, 3))
    Tr = Tr.reshape((3, 4))

    T = np.zeros((1,4))
    T[0,3] = 1

    P2 = np.vstack((P2, T))
    Tr = np.vstack((Tr, T))

    T2 = np.zeros((4,1))
    T2[3,0] = 1
    R0 = np.hstack((R0, T2))

    assert Tr.shape == (4, 4)
    assert R0.shape == (4, 4)
    assert P2.shape == (4, 4)

    point = point.transpose((1, 0))

    point = np.vstack((point, np.ones(point.shape[1])))

    mat1 =  np.dot(P2, R0)
    mat2 = np.dot(mat1, Tr)
    img_cor = np.dot(mat2, point)

    img_cor = img_cor/img_cor[2]

    return img_cor

def camera_to_lidar_cnr(pts_3D, P):
    """
    convert camera corners to lidar corners
    """
    if pts_3D.shape[1] == 24:
        print "aaaaa"
        pts_3D = pts_3D.reshape((3, 8))

    pts_3D = np.vstack((pts_3D, np.ones(8)))

    assert pts_3D.shape == (4, 8)

    # R = np.linalg.inv(P[:, :3])
    # # T = -P[:, 3].reshape((3, 1))
    # T = np.zeros((3, 1))
    # T[0] = -P[1,3]
    # T[1] = -P[2,3]
    # T[2] = P[0,3]
    # RT = np.hstack((R, T))
    #
    # lidar_corners = np.dot(RT, pts_3D)
    # lidar_corners = lidar_corners[:3,:]
    #
    # print "raw_RT", RT


    ### rewrite by Jack Qian
    T = np.zeros((1,4))
    T[0,3] = 1
    new_P = np.vstack((P, T))
    new_P_inv = np.linalg.inv(new_P)
    # print "new_P:", new_P
    # print "new_P_inv: ", new_P_inv

    lidar_corners = np.dot(new_P_inv, pts_3D)
    lidar_corners = lidar_corners[:3,:]

    return lidar_corners.reshape(-1, 24)

def lidar_cnr_to_camera(pts_cnr, P):
    """
    convert lidar corners to camera corners
    """
    pts_cnr = pts_cnr.reshape((3, 8))

    P = P.astype(np.float32).reshape(3, 4)
    p_new = np.vstack((P,np.zeros(P.shape[1])))
    p_new[3, 3] = 1

    pts_cnr = np.vstack((pts_cnr, np.ones(8)))

    assert pts_cnr.shape == (4, 8)

    # R = np.linalg.inv(P[:, :3])
    # # T = -P[:, 3].reshape((3, 1))
    # T = np.zeros((3, 1))
    # T[0] = -P[1,3]
    # T[1] = -P[2,3]
    # T[2] = P[0,3]
    #
    # RT = np.hstack((R, T))
    P = P.reshape((3, 4))
    RT = P

    lidar_corners = np.dot(p_new, pts_cnr)
    lidar_corners = lidar_corners[:3,:]

    return lidar_corners.reshape(-1, 24)


def corners_to_img(corners, Tr, R0, P2):
    
    Tr = Tr.reshape((3, 4))
    R0 = R0.reshape((4, 3))
    P2 = P2.reshape((3, 4))
    
    if 24 in corners.shape:
        corners = corners.reshape((3, 8))

    # R0 = np.vstack((R0, np.zeros(3)))
    # print corners.shape
    corners = np.vstack((corners, np.zeros(8)))

    # print(corners.shape)
    img_cor = reduce(np.dot, [P2, R0, Tr, corners])

    return img_cor

def corners_to_boxes(corners):
    corners = corners.reshape(3,8)

    xyz = corners.mean(1)

    ww = (corners[0, 0] - corners[0, 1]) * (corners[0, 0] - corners[0, 1]) + (corners[2, 0] - corners[2, 1]) * (corners[2, 0] - corners[2, 1])

    w = np.sqrt(ww)


    ll = (corners[0, 1] - corners[0, 2]) * (corners[0, 1] - corners[0, 2]) + (corners[2, 1] - corners[2, 2]) * (corners[2, 1] - corners[2, 2])
    l = np.sqrt(ll)
    h = np.abs(corners[1, 0] - corners[1, 4])


    dx = corners[0, 0] - corners[0, 1]
    dy = corners[2, 0] - corners[2, 1]
    ry = np.arctan2(dx, dy)

    xyz[1] = xyz[1]+h/2

    return [h,w,l,xyz[0],xyz[1],xyz[2],ry]

if __name__ == '__main__':
    # P = np.array([6.927964000000e-03, -9.999722000000e-01, -2.757829000000e-03,
    #               -2.457729000000e-02, -1.162982000000e-03, 2.749836000000e-03,
    #               -9.999955000000e-01, -6.127237000000e-02, 9.999753000000e-01,
    #               6.931141000000e-03, -1.143899000000e-03, -3.321029000000e-01]).astype(np.float32).reshape((3, 4))
    # camera = [1.84, 1., 8.41, 5.78, 1.90, 2.72]
    # lidar = camera_to_lidar(camera, P)
    # print lidar
    # corners = lidar_to_corners_single(lidar)
    # corners = corners.reshape((3, 8))
    # # print(lidar)
    # print(corners)

    GT_boxes3D_corners = np.array([1.11426134e+01 ,  1.12146721e+01 ,  1.53606148e+01 ,  1.52885561e+01,
                        1.11193733e+01 ,  1.11914320e+01 ,  1.53373756e+01 ,  1.52653170e+01,
                        - 1.67702079e+00 ,  5.14763035e-02, - 1.21444792e-01, - 1.84994197e+00,
                        - 1.67816389e+00 ,  5.03332280e-02, - 1.22587867e-01, - 1.85108495e+00,
                        - 1.58645678e+00, - 1.58413136e+00, - 1.52288032e+00, - 1.52520561e+00,
                        - 1.66291781e-02, - 1.43038025e-02,   4.69473116e-02,   4.46219370e-02]).astype(np.float32).reshape(3, 8)

    GT_boxes3D_camera_corners = np.array([  1.76189673,   0.03399043,   0.2381033,    1.96600962,   1.76189673,
                               0.03399043,   0.2381033,    1.96600962,   1.75,         1.75,         1.75,
                               1.75,         0.17999995,   0.17999995,   0.17999995,   0.17999995,
                               11.10496712,  11.19005585,  15.33503342,  15.24994469,  11.10496712,
                               11.19005585,  15.33503342,  15.24994469]).astype(np.float32).reshape(3, 8)

    Tr = np.array([7.533745000000e-03, -9.999714000000e-01, -6.166020000000e-04, -4.069766000000e-03,
                    1.480249000000e-02, 7.280733000000e-04, -9.998902000000e-01, -7.631618000000e-02,
                    9.998621000000e-01, 7.523790000000e-03, 1.480755000000e-02, -2.717806000000e-01]).astype(np.float32).reshape(3, 4)

    print lidar_cnr_to_camera(GT_boxes3D_corners,Tr)
    corners_to_boxes(GT_boxes3D_camera_corners)