import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)

class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3, zc=300):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.zc = zc
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 6))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,d],[x1,y1,x2,y2,d],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    Z = []
    X = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:], zc=self.zc)
        self.trackers.append(trk)

    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          Z.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
          X.append(trk.kf.x.reshape(1,-1))
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(Z)>0):
      return np.concatenate(Z), np.concatenate(np.array(X))
    return np.empty((0,7)), np.empty((0,11))

class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox, zc=300, img_shape=(1280,720)):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.zc = zc
    self.img_shape = img_shape

    self.kf = KalmanFilter(dim_x=11, dim_z=6)
    # [x, y, z, yaw, area, ratio, vx, vy, vz, vyaw, varea]
    # state transition function (process)
    self.kf.F = np.array([[1, 0, 0, 0, 0, 0, 1/30, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0, 1/30, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0, 0, 1/30, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1/30],
                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    
    # measurement function
    self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])

    # state variance
    # self.kf.P[6:,6:] *= 1000. #give high uncertainty to the unobservable initial velocities
    # self.kf.P *= 10.
    self.kf.P = np.diag([100, 40000, 400, 10000, 10000, 10, 25, 10, 100, 100, 20000])

    # process white noise
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[6:,6:] *= 0.01

    # measurement noise
    # self.kf.R[4:,4:] *= 10.
    self.kf.R = np.diag([10, 100, 25, 30000, 100, 10])

    # initial state
    self.kf.x[:6] = convert_bbox_to_z(bbox, zc=self.zc, img_shape=self.img_shape)

    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox, zc=self.zc, img_shape=self.img_shape))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[10]+self.kf.x[4])<=0):
      self.kf.x[4] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x, zc=self.zc, img_shape=self.img_shape))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x, zc=self.zc, img_shape=self.img_shape)

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,6),dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  


def convert_bbox_to_z(bbox, zc, img_shape):
  """
  Takes a bounding box in the form [x1,y1,x2,y2,yaw], zc (camera constant), and img_shape (w,h), and returns z in the form
    [x,y,z,yaw,s,r] where x,y,z is the position of the object, yaw is the orientation angle, s is the scale/area, and r is ratio
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  xc = bbox[0] + w/2. - img_shape[0]/2
  yc = bbox[1] + h/2.
  d = 1.5 * (np.sqrt(zc**2 + (img_shape[1]/2 - yc)**2)) / h

  x = d * np.sin(np.arctan(xc / zc))  # d * sin(tan-1(xc / zc))
  y = yc
  z = d * np.cos(np.arctan(xc / zc))  # d * math.cos(tan-1(xc / zc))
  yaw = bbox[4]
  s = w * h    #scale is just area
  r = w / float(h)

  return np.array([x, y, z, yaw, s, r]).reshape((6, 1))


def convert_x_to_bbox(x, zc, img_shape, score=None):
  """
  Takes a bounding box in the centre form [x,y,z,yaw,s,r] and returns it in the form
    [x1,y1,x2,y2,yaw,d] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[4] * x[5])
  h = x[4] / w
  # TODO: compute y1 y2 (currently it is 0)
  d = np.sqrt(x[0] ** 2 + x[2] ** 2)
  xc = img_shape[0]/2 + zc * np.tan(np.arcsin(x[0] / d))
  yc = x[1]
  yaw = x[3]

  if(score==None):
    return np.array([xc-w/2., yc-h/2, xc+w/2., yc+h/2, yaw, d]).reshape((1,6))
  else:
    return np.array([xc-w/2., yc-h/2, xc+w/2., yc+h/2, yaw, d, score]).reshape((1,7))
