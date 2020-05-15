
import os
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import numpy as np


data_root = '/media/ferdyan/LocalDiskE/Kitti/object/training/'


imgIdx = 20
dataname = 'dll'
date = '_'.join(dataname.split('_')[:3])
dictmap = {'left':'image_2/', 'right':'image_3/'}

cam_type = dictmap[['left','right'][1]]
cam_idx = 2 if cam_type=='left' else 3

file_root = os.path.join(data_root, date, dataname, cam_type)
cam_dir = os.path.join(data_root, date)
cam_txt = os.path.join(cam_dir, 'calib_velo_to_cam.txt' )

data_dir = "/media/ferdyan/LocalDiskE/Kitti/object/training/"
img_path = os.path.join(data_dir, "image_2/{:06d}.png")
depth_gt_path01 = os.path.join(data_dir, "v2/{:06d}.bin")
cam_dir = os.path.join(data_dir, "calib/{:06d}.txt")


# ====================================================================
import matplotlib.pyplot as plt
# COLOR: https://blog.csdn.net/guduruyu/article/details/60868501
def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename.format(frame_num), dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def project_velo_to_cam(calib_dir, cam_idx):
    # ====
    print("sini")
    velo_calib = read_calib_file(calib_dir.format(frame_num))

    R_awal = velo_calib['Tr_velo_to_cam']
    # R = np.array(R_awal[3:12]).reshape(3, 3)
    # t = np.array(R_awal[0:3]).reshape(3, 1)
    # R = np.array(R_awal[0:9]).reshape(3, 3)
    # t = np.array(R_awal[9:12]).reshape(3, 1)
    # P_velo2cam_ref = np.vstack((np.hstack([R, t]), np.array([0., 0., 0., 1.])))  # velo2ref_cam

    P_velo2cam_ref = np.zeros((4,4))
    P_velo2cam_ref[3,3] = 1
    P_velo2cam_ref[:3,:4] = velo_calib['Tr_velo_to_cam'].reshape(3,4)

    # ====
    cam_calib = read_calib_file(calib_dir.format(frame_num))
    R0_rect = cam_calib['R0_rect'].reshape(3, 3)  # ref_cam2rect
    R_ref2rect = np.eye(4)
    R_ref2rect[:3, :3] = R0_rect
    # ====
    P_rect2cam2 = cam_calib['P2'].reshape((3, 4)) ##
    # ====
    # print("R = ", R)
    # print("t = ", t)
    # print("R0_rect = ", R0_rect)
    # print("P_rect2cam2 = ", P_rect2cam2)
    
    proj_mat = P_rect2cam2 @ R_ref2rect @ P_velo2cam_ref
    return proj_mat

def project_to_image(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        pts_3d:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    """
    num_pts = points.shape[1]
    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    # pts3d = points[:, points[-1, :] > 0].copy()
    # pts3d[-1, :] = 1
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    return points[:2, :]

def render_lidar_on_image(lidar_path, img, calib_dir, cam_idx, img_width, img_height):
    pts_velo = load_velo_scan(lidar_path)[:, :3]

    print("pts_velo.shape = ", pts_velo.shape)
    # pts3d = pts_velo[:, pts_velo[-1, :] > 0].copy()
    # pts3d[-1, :] = 1


    # projection matrix (project from velo2cam2)
    proj_velo2cam = project_velo_to_cam (calib_dir, cam_idx)

    print("proj_velo2cam = ", proj_velo2cam)
    print("proj_velo2cam.shape = ", proj_velo2cam.shape)

    # apply projection
    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam)

    #print("pts_2d = ", pts_2d.shape)

    # Filter lidar points to be within image FOV   
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pts_velo[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]

    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    imgfov_pc_cam = proj_velo2cam @ imgfov_pc_velo.transpose()
    
    vis_data = imgfov_pc_pixel
    vis_field = vis_data.shape[1]
    depth = imgfov_pc_cam[2, :vis_field ]
    #color = cmap[int(640.0 /depth), :]
    
    x = np.array([int(np.round(vis_data[0, i])) for i in range(vis_field )])
    y = np.array([int(np.round(vis_data[1, i])) for i in range(vis_field )])
    """
    plt.title("Lidar visualize")
    plt.set_cmap('hsv')#('gist_rainbow')#('jet')#('brg_r')

    plt.scatter(x, y, c=depth, alpha=0.15, s=10,zorder=2)#linewidths=1)
    cbar= plt.colorbar()
    cbar.set_label("(m)", labelpad=+1)
    plt.imshow(img,zorder=1)
    """

    fig, (ax1) = plt.subplots(1,1)
    #fig.colorbar()
    
    
    ax1.set_title("Alpha value of 0.5")
    #ax1.set_axis_off()

    ax1.scatter(x, y, c=depth, alpha=0.15, s=10, cmap='hsv')
    ax1.imshow(img)
    
    #ax2.scatter(x, y, c=depth, alpha=0.15, s=10, cmap='hsv')
    #ax2.set_title("Alpha value of 1")
    
    plt.show()
    return img

frame_num = 78

rgb = cv2.cvtColor(cv2.imread(img_path.format(frame_num)), cv2.COLOR_BGR2RGB)
img_height, img_width, img_channel = rgb.shape

print("img_height = ", img_height)
print("img_width = ", img_width)

render_lidar_on_image(depth_gt_path01, rgb, cam_dir, cam_idx, img_width, img_height)
