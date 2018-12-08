import os
import numpy as np
import cv2


def read_intrinsics_raw(dataset_dir, date, cam_id):
    calib_file = os.path.join(dataset_dir, date, 'calib_cam_to_cam.txt')
    filedata = read_raw_calib_file(calib_file)
    P_rect = np.reshape(filedata['P_rect_' + cam_id], (3, 4))
    intrinsics = P_rect[:3, :3]
    return intrinsics


def read_raw_calib_file(filepath):
    # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def read_file_data(data_root, filename):
    date, drive, cam, _, frame_id = filename.split("/")
#         camera_id = filename[-1]   # 2 is left, 3 is right
    vel = '{}/{}/velodyne_points/data/{}.bin'.format(date, drive, frame_id[:10])
    img_file = os.path.join(data_root, filename)
    num_probs = 0

    if os.path.isfile(img_file):
        gt_file = os.path.join(data_root, vel)
        gt_calib = os.path.join(data_root, date)
        im_size = cv2.imread(img_file).shape[:2]
        cam_id = cam[-2:]
        return gt_file, gt_calib, im_size, cam_id
    else:
        num_probs += 1
        print('{} missing'.format(img_file))
        return [], [], [], []


def read_odom_calib_file(filepath, cid=2):
    """Read in a calibration file and parse into a dictionary."""
    with open(filepath, 'r') as f:
        calib = f.readlines()

    def parse_line(L, shape):
        data = L.split()
        data = np.array(data[1:]).reshape(shape).astype(np.float32)
        return data

    proj_c2p = parse_line(calib[cid], shape=(3, 4))
    proj_v2c = parse_line(calib[-1], shape=(3, 4))
    filler = np.array([0, 0, 0, 1]).reshape((1, 4))
    proj_v2c = np.concatenate((proj_v2c, filler), axis=0)
    return proj_c2p, proj_v2c


def scale_intrinsics(mat, sx, sy):
    out = np.copy(mat)
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out
