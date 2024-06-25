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

class SortLane(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=60):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 4))):
    """
    Params:
      dets - a numpy array of detections in the format [[p1,p2,p3,p4],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 4)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    Z = []
    X = []

    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 4))
    to_del = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3]]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanLaneTracker(dets[i,:])
        self.trackers.append(trk)

    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
          Z.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
          X.append(trk.kf.x.reshape(1,-1))
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(Z)>0):
      return np.concatenate(Z), np.concatenate(np.array(X))
    return np.empty((0,4)), np.empty((0,8))

class KalmanLaneTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self, bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model

    self.kf = KalmanFilter(dim_x=8, dim_z=4)
    # [p1, p2, p3, p4, vp1, vp2, vp3, vp4]
    # state transition function (process)
    self.kf.F = np.array([[1, 0, 0, 0, 1/30, 0, 0, 0],
                          [0, 1, 0, 0, 0, 1/30, 0, 0],
                          [0, 0, 1, 0, 0, 0, 1/30, 0],
                          [0, 0, 0, 1, 0, 0, 0, 1/30],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1]])
    
    # measurement function
    self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0]])

    # state variance
    # self.kf.P[6:,6:] *= 1000. #give high uncertainty to the unobservable initial velocities
    # self.kf.P *= 10.
    self.kf.P = np.diag([225, 225, 225, 225, 1000, 1000, 1000, 1000])

    # process white noise
    # self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    # measurement noise
    # self.kf.R[4:,4:] *= 10.
    self.kf.R = np.diag([1000, 1000, 1000, 1000])

    # initial state
    self.kf.x[:4] = convert_bbox_to_z(bbox)

    self.time_since_update = 0
    self.id = KalmanLaneTracker.count
    KalmanLaneTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self, bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.kf.predict()
    self.age += 1
    # if(self.time_since_update > 0):
    #   self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def associate_detections_to_trackers(detections, trackers, iou_threshold = 60):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,4),dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix < iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(iou_matrix)
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
    if(iou_matrix[m[0], m[1]] > iou_threshold):
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
  From SORT: Computes IOU between two bboxes in the form [p1,p2,p3,p4]
  """
  bb_test = np.expand_dims(bb_test, 1)
  bb_gt = np.expand_dims(bb_gt, 0)

  result = np.abs(bb_test[..., 0] - bb_gt[..., 0])
  for i in range(1, 4):
    result += np.abs(bb_test[..., i] - bb_gt[..., i])

  return result

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [p1,p2,p3,p4], and returns z in the form
    [p1,p2,p3,p4] where each p is the x coordinate
    the aspect ratio
  """

  return np.array(bbox).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
  """
  Takes a bounding box in the centre form [p1,p2,p3,p4,vp1,vp2,vp3,vp4] and returns it in the form
    [p1,p2,p3,p4] where each p is the x coordinate of point
  """

  if(score==None):
    return np.array([x[0], x[1], x[2], x[3]]).reshape((1,4))
  else:
    return np.array([x[0], x[1], x[2], x[3], score]).reshape((1,5))
