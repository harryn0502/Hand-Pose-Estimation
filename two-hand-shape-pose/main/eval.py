import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys,os,cv2,math,glob,scipy
import scipy.io as sio
#import tensorflow as tf
from PIL import Image

#from config import config
# from eval_util import *


class Evaluation(object):
    def eval_3dpck(self,gt_joints,pred_joints,joints_vis = None):        
        num_samples = gt_joints.shape[0]
        num_joints  = gt_joints.shape[1]

        if joints_vis is None:
            joints_vis = np.ones([num_samples,num_samples])
        util = EvalUtil(num_kp=num_joints)
        for i in range(num_samples):
            util.feed(gt_joints[i,:,:],joints_vis[i,:],pred_joints[i,:,:])

        # Output results
        mean, median, auc, pck_curve_all, threshs = util.get_measures(0, 50, 200)
        print('Average mean EPE: %.3f mm' %mean)
        print('Average median EPE: %.3f mm' %median)
        print('Area under curve between 0mm - 50mm: %.3f' % auc)

        pck_curve_all1, threshs1 = pck_curve_all[80:], threshs[80:]
        auc_subset1 = calc_auc(threshs1, pck_curve_all1)
        print('Area under curve between 20mm - 50mm: %.3f' % auc_subset1)

        pck_curve_all2, threshs2 = pck_curve_all[:120], threshs[:120]
        auc_subset2 = calc_auc(threshs2, pck_curve_all2)
        print('Area under curve between 0mm - 30mm: %.3f' % auc_subset2)

        mean, median, auc, pck_curve_all, threshs = util.get_measures(0, 100, 200)
        print('Area under curve between 0mm - 100mm: %.3f' % auc)

        return pck_curve_all, threshs, mean, median, auc, auc_subset1, auc_subset2

    def eval_2dpck(self,gt_joints,pred_joints,joints_vis = None):
        num_samples = gt_joints.shape[0]
        num_joints  = gt_joints.shape[1]
        if joints_vis is None:
            joints_vis = np.ones([num_samples,num_samples])

        util = EvalUtil(num_kp=num_joints)
        for i in range(num_samples):
            util.feed(gt_joints[i,:,:],joints_vis[i,:],pred_joints[i,:,:])

        # Output results
        mean, median, auc, pck_curve_all, threshs = util.get_measures(0, 30, 700)
        print('Average mean EPE: %.3f px' %mean)
        print('Average median EPE: %.3f px' %median)
        print('Area under curve between 0px - 30px: %.3f' % auc)
        return pck_curve_all, threshs, mean, median, auc, auc, auc

class EvalUtil:
    """ Util class for evaluation networks.
    """
    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_vis, keypoint_pred):
        """ Used to feed data to the class. Stores the euclidean distance between gt and pred, when it is visible. """
        keypoint_gt = np.squeeze(keypoint_gt)
        keypoint_pred = np.squeeze(keypoint_pred)
        keypoint_vis = np.squeeze(keypoint_vis).astype('bool')

        assert len(keypoint_gt.shape) == 2
        assert len(keypoint_pred.shape) == 2
        assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))

        num_kp = keypoint_gt.shape[0]
        for i in range(num_kp):
            if keypoint_vis[i]:
                self.data[i].append(euclidean_dist[i])

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None
        
        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)

        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()
        
        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            
            mean, median = self._get_epe(part_id)
            # print len(self.data[part_id]),mean
            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

        return epe_mean_all, epe_median_all, auc_all, pck_curve_all, thresholds
    
    def plot_pck(self,threshs,pck_curve_all,auc_subset):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(threshs, pck_curve_all, label='Ours (AUC=%.3f)' % auc_subset)
        ax.set_xlabel('threshold in mm')
        ax.set_ylabel('PCK')
        plt.legend(loc='lower right')
        plt.xlim(threshs[0],threshs[-1])
        fig.savefig('pck.jpg')

def calc_auc(x, y):
    """ Given x and y values it calculates the approx. integral and normalizes it: area under curve"""
    integral = np.trapz(y, x)
    norm = np.trapz(np.ones_like(y), x)

    return integral / norm
