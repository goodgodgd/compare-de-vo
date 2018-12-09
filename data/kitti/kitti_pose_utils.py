# Some of the code are from the TUM evaluation toolkit:
# https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#absolute_trajectory_error_ate
import os
import math
import numpy as np
import functools


def save_pose_result(pose_seqs, output_root, modelname, frames, seq_length):
    assert os.path.isdir(output_root), "[ERROR] dir not found: {}".format(output_root)
    save_path = os.path.join(output_root, modelname, "pose")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    sequences = []
    for i, (poseseq, frame) in enumerate(zip(pose_seqs, frames)):
        seq_id, frame_id = frame.split(" ")
        if seq_id not in sequences:
            sequences.append(seq_id)
            if not os.path.isdir(os.path.join(save_path, seq_id)):
                os.makedirs(os.path.join(save_path, seq_id))

        half_seq = (seq_length - 1) // 2
        filename = os.path.join(save_path, seq_id, "{:06d}.txt".format(int(frame_id)-half_seq))
        np.savetxt(filename, poseseq, fmt="%06f")
    print("pose results were saved!!")


def compute_pose_error(gt_sseq, pred_sseq):
    gt_sseq, pred_sseq, ali_inds = align_pose_seq(gt_sseq, pred_sseq)

    assert gt_sseq.shape == pred_sseq.shape, "after alignment, gt:{}, {} == pred:{}, {}"\
        .format(gt_sseq.shape[0], gt_sseq.shape[1], pred_sseq.shape[0], pred_sseq.shape[1])
    seq_len = gt_sseq.shape[0]
    err_result = []
    for si in range(1, seq_len):
        te, re = pose_diff(gt_sseq[si], pred_sseq[si])
        err_result.append([ali_inds[si], te, re])
        assert ali_inds[si] > 0, "{} {}".format(si, ali_inds)
    return err_result


def align_pose_seq(gt_sseq, pred_sseq, max_diff=0.01):
    assert abs(gt_sseq[0, 0] - pred_sseq[0, 0]) < max_diff, \
        "different initial time: {}, {}".format(gt_sseq[0, 0], pred_sseq[0, 0])

    gt_times = gt_sseq[:, 0].tolist()
    pred_times = pred_sseq[:, 0].tolist()
    potential_matches = [(abs(gt - pt), gt, gi, pt, pi)
                         for gi, gt in enumerate(gt_times)
                         for pi, pt in enumerate(pred_times)
                         if abs(gt - pt) < max_diff]
    potential_matches.sort()
    matches = []
    for diff, gt, gi, pt, pi in potential_matches:
        if gt in gt_times and pt in pred_times:
            gt_times.remove(gt)
            pred_times.remove(pt)
            matches.append((gi, pi))
    matches.sort()
    aligned_inds = [gi for gi, pi in matches]

    if len(matches) < 2:
        raise ValueError("aligned poses are {} from {}".format(len(matches), len(potential_matches)))

    aligned_gt = []
    aligned_pred = []
    for gi, pi in matches:
        aligned_gt.append(gt_sseq[gi])
        aligned_pred.append(pred_sseq[pi])
    aligned_gt = np.array(aligned_gt)
    aligned_pred = np.array(aligned_pred)
    return aligned_gt, aligned_pred, aligned_inds


def pose_diff(gt_pose, pred_pose):
    # Optimize the scaling factor
    assert np.sum(pred_pose[1:4] ** 2) > 0.00001, \
        "invalid scale division {}".format(np.sum(pred_pose[1:4] ** 2))
    scale = np.sum(gt_pose[1:4] * pred_pose[1:4]) / np.sum(pred_pose[1:4] ** 2)
    # translational error
    alignment_error = pred_pose[1:4] * scale - gt_pose[1:4]
    trn_rmse = np.sqrt(np.sum(alignment_error ** 2))
    # rotation matrices
    gt_rmat = quat2mat(gt_pose[4:])
    pred_rmat = quat2mat(pred_pose[4:])
    # relative rotation
    rel_rmat = np.matmul(np.transpose(gt_rmat), pred_rmat)
    rel_quat = rot2quat(rel_rmat)
    rot_diff = abs(np.arccos(rel_quat[0]))
    return trn_rmse, rot_diff


def rot2quat(R):
    rz, ry, rx = mat2euler(R)
    qw, qx, qy, qz = euler2quat(rz, ry, rx)
    return qw, qx, qy, qz


def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion
    https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/quaternions.py
    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows non-unit
    quaternions.

    References
    ----------
    Algorithm from
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < 1e-8:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])


def mat2euler(M, cy_thresh=None, seq='zyx'):
    '''
    Taken From: http://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py
    Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
     threshold below which to give up on straightforward arctan for
     estimating x rotation.  If None (default), estimate from
     precision of input.
    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
     Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
    [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
    [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
    [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    with the obvious derivations for z, y, and x
     z = atan2(-r12, r11)
     y = asin(r13)
     x = atan2(-r23, r33)
    for x,y,z order
    y = asin(-r31)
    x = atan2(r32, r33)
    z = atan2(r21, r11)
    Problems arise when cos(y) is close to zero, because both of::
     z = atan2(cos(y)*sin(z), cos(y)*cos(z))
     x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    will be close to atan2(0, 0), and highly unstable.
    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:
    See: http://www.graphicsgems.org/
    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if seq=='zyx':
        if cy > cy_thresh: # cos(y) not close to zero, standard form
            z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = math.atan2(r13,  cy) # atan2(sin(y), cy)
            x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else: # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = math.atan2(r21,  r22)
            y = math.atan2(r13,  cy) # atan2(sin(y), cy)
            x = 0.0
    elif seq=='xyz':
        if cy > cy_thresh:
            y = math.atan2(-r31, cy)
            x = math.atan2(r32, r33)
            z = math.atan2(r21, r11)
        else:
            z = 0.0
            if r31 < 0:
                y = np.pi/2
                x = atan2(r12, r13)
            else:
                y = -np.pi/2
    else:
        raise Exception('Sequence not recognized')
    return z, y, x


def euler2mat(z=0, y=0, x=0, isRadian=True):
    ''' Return matrix for rotations around z, y and x axes
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
         Rotation angle in radians around z-axis (performed first)
    y : scalar
         Rotation angle in radians around y-axis
    x : scalar
         Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    M : array shape (3,3)
         Rotation matrix giving same rotation as for given angles
    Examples
    --------
    >>> zrot = 1.3 # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True
    The output rotation matrix is equal to the composition of the
    individual rotations
    >>> M1 = euler2mat(zrot)
    >>> M2 = euler2mat(0, yrot)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True
    You can specify rotations by named arguments
    >>> np.all(M3 == euler2mat(x=xrot))
    True
    When applying M to a vector, the vector should column vector to the
    right of M.  If the right hand side is a 2D array rather than a
    vector, then each column of the 2D array represents a vector.
    >>> vec = np.array([1, 0, 0]).reshape((3,1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
    >>> vecs2 = np.dot(M, vecs)
    Rotations are counter-clockwise.
    >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
    True
    Notes
    -----
    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
    '''

    if not isRadian:
        z = ((np.pi)/180.) * z
        y = ((np.pi)/180.) * y
        x = ((np.pi)/180.) * x
    assert z>=(-np.pi) and z < np.pi, 'Inapprorpriate z: %f' % z
    assert y>=(-np.pi) and y < np.pi, 'Inapprorpriate y: %f' % y
    assert x>=(-np.pi) and x < np.pi, 'Inapprorpriate x: %f' % x    

    Ms = []
    if z:
            cosz = math.cos(z)
            sinz = math.sin(z)
            Ms.append(np.array(
                            [[cosz, -sinz, 0],
                             [sinz, cosz, 0],
                             [0, 0, 1]]))
    if y:
            cosy = math.cos(y)
            siny = math.sin(y)
            Ms.append(np.array(
                            [[cosy, 0, siny],
                             [0, 1, 0],
                             [-siny, 0, cosy]]))
    if x:
            cosx = math.cos(x)
            sinx = math.sin(x)
            Ms.append(np.array(
                            [[1, 0, 0],
                             [0, cosx, -sinx],
                             [0, sinx, cosx]]))
    if Ms:
            return functools.reduce(np.dot, Ms[::-1])
    return np.eye(3)


def euler2quat(z=0, y=0, x=0, isRadian=True):
    ''' Return quaternion corresponding to these Euler angles
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
         Rotation angle in radians around z-axis (performed first)
    y : scalar
         Rotation angle in radians around y-axis
    x : scalar
         Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    quat : array shape (4,)
         Quaternion in w, x, y z (real, then vector) format
    Notes
    -----
    We can derive this formula in Sympy using:
    1. Formula giving quaternion corresponding to rotation of theta radians
         about arbitrary axis:
         http://mathworld.wolfram.com/EulerParameters.html
    2. Generated formulae from 1.) for quaternions corresponding to
         theta radians rotations about ``x, y, z`` axes
    3. Apply quaternion multiplication formula -
         http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
         formulae from 2.) to give formula for combined rotations.
    '''
  
    if not isRadian:
        z = ((np.pi)/180.) * z
        y = ((np.pi)/180.) * y
        x = ((np.pi)/180.) * x
    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return np.array([
                     cx*cy*cz - sx*sy*sz,
                     cx*sy*sz + cy*cz*sx,
                     cx*cz*sy - sx*cy*sz,
                     cx*cy*sz + sx*cz*sy])


def pose_vec_to_mat(vec):
    tx = vec[0]
    ty = vec[1]
    tz = vec[2]
    trans = np.array([tx, ty, tz]).reshape((3,1))
    rot = euler2mat(vec[5], vec[4], vec[3])
    Tmat = np.concatenate((rot, trans), axis=1)
    hfiller = np.array([0, 0, 0, 1]).reshape((1,4))
    Tmat = np.concatenate((Tmat, hfiller), axis=0)
    return Tmat


def format_pose_seq_TUM(poses, times):
    if isinstance(poses, list):
        tum_poses = format_pose_list(poses, times)
    elif isinstance(poses, np.ndarray):
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        tum_poses = format_npy_array(poses, times)
    else:
        tum_poses = None
    return tum_poses


def format_pose_list(poses, times):
    # poses: list of [tx, ty, tz, rx, ry, rz]
    pose_seq = []
    # First frame as the origin
    first_pose = pose_vec_to_mat(poses[0])
    for pose, time in zip(poses, times):
        this_pose = pose_vec_to_mat(pose)
        # this_pose = np.dot(this_pose, np.linalg.inv(first_pose))
        this_pose = np.dot(first_pose, np.linalg.inv(this_pose))
        tx = this_pose[0, 3]
        ty = this_pose[1, 3]
        tz = this_pose[2, 3]
        rot = this_pose[:3, :3]
        qw, qx, qy, qz = rot2quat(rot)
        pose = np.array([time, tx, ty, tz, qx, qy, qz, qw])
        pose_seq.append(pose)
    pose_seq = np.array(pose_seq)
    return pose_seq


def format_npy_array(poses, times):
    pose_list = []
    for i in range(poses.shape[0]):
        pose_list.append(poses[i, :])
    return format_pose_list(pose_list, times.tolist())
