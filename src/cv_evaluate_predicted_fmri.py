# coding: utf-8

"""
DNN Feature encoding (corss-validation) - feature evaluation script.

encoding prediction 自体は VoxelData 全体で行う
その中身を解析ターゲットのROIごとに分けて，profile correlationなど計算するようにしたやつ
"""

from typing import Dict, List, Optional
from itertools import product
import os
import hdf5storage
import numpy as np

import sys
sys.path.insert(0, "./bdpy/")
import bdpy
print(bdpy.__file__)

from bdpy.pipeline.config import init_hydra_cfg
from bdpy.dataform import SQLite3KeyValueStore
from bdpy.evals.metrics import profile_correlation, pattern_correlation, pairwise_identification


# Main #######################################################################


class ResultsStore(SQLite3KeyValueStore):
    """Results store for feature decoding evaluation."""
    pass


def cv_evaluate_predicted_fmri(
    encoded_fmri_dir: str,
    encoder_dir: str,
    fmri: Dict[str, List[str]],
    training_rois: Dict[str, str],
    rois: Dict[str, str],
    features: List[str],
    cv_folds: Optional[List[Dict[str, List]]],
    label_key: str,
    output_file_pooled: str = './evaluation.db',
    output_file_fold: str = './evaluation_fold.db',
    average_sample: bool = True,
):
    _features = features[::-1]  # Start training from deep layers
    _subjects = list(fmri.keys())
    _training_rois = list(training_rois.keys())
    _rois = list(rois.keys())
    np.random.shuffle(_features)
    np.random.shuffle(_subjects)
    np.random.shuffle(_training_rois)
    np.random.shuffle(_rois)

    # Display information
    print('Subjects:        {}'.format(_subjects))
    print('Training ROIs:   {}'.format(_training_rois))
    print('Evaluation ROIs: {}'.format(_rois))
    print('Layers:          {}'.format(_features))
    print('')
    print('Encoded fmri dir: {}'.format(encoded_fmri_dir))
    print('True features (Test): {}'.format(fmri))
    print('')
    print('')

    # Metrics ################################################################
    metrics = ['profile_correlation',
               # 'pattern_correlation', 'identification_accuracy', 'identification_accuracy_predictedbase'
               ]
    pooled_operation = {
        "profile_correlation": "mean",
        # "pattern_correlation": "concat",
        # "identification_accuracy": "concat",
        # "identification_accuracy_predictedbase": "concat",
    }

    # Evaluating encoding performances #######################################

    if os.path.exists(output_file_fold):
        print('Loading {}'.format(output_file_fold))
        results_db = ResultsStore(output_file_fold)
    else:
        print('Creating new evaluation result store')
        keys = ["layer", "subject", "roi", "fold", "metric"]
        results_db = ResultsStore(output_file_fold, keys=keys)

    if 'name' in cv_folds[0]:
        cv_labels = ['cv-{}'.format(cv['name']) for cv in cv_folds]
    else:
        cv_labels = ['cv-fold{}'.format(icv + 1) for icv in range(len(cv_folds))]

    finish_fold = True
    for subject in _subjects:
        for train_roi in _training_rois:
            for feature, fold in product(_features, cv_labels):
                for eval_roi in _rois:
                    for metric in metrics:
                        if not results_db.exists(layer=feature, subject=subject, roi=eval_roi,
                                                 fold=fold, metric=metric):
                            finish_fold = False
                            break

    if not finish_fold:
        for subject in _subjects:
            print("Subject:", subject)
            # Loading data ###########################################################
            # True fmri data
            print("Load bdata...")
            bdata = bdpy.vstack([bdpy.BData(f) for f in fmri[subject]], successive=['Session', 'Run', 'Block'])
            print("Done.")

            for train_roi in _training_rois:
                print("Training ROI:", train_roi)
                # Training ROI's observed brain activity
                observed_activity, train_roi_selector = bdata.select(training_rois[train_roi], return_index=True)
                observed_labels = np.asarray(bdata.get_label(label_key))
                if average_sample:
                    unique_observed_labels = np.unique(observed_labels)
                    observed_activity_list = []
                    for label in unique_observed_labels:
                        observed_activity_list.append(np.nanmean(observed_activity[observed_labels == label, :],
                                                                 axis=0))
                    observed_activity = np.vstack(observed_activity_list)
                    observed_labels = unique_observed_labels

                for feature, fold in product(_features, cv_labels):
                    print("Feature:        {}".format(feature))
                    print("Fold:           {}".format(fold))

                    fmri_dir = os.path.join(encoded_fmri_dir, feature, subject,
                                            train_roi, fold, 'encoded_fmri')
                    # model_dir = os.path.join(encoder_dir, feature, subject,
                    #                          train_roi, fold, 'model')

                    # Training ROI's encoded brain activity
                    encoded_activity_list = []
                    encoded_labels = []
                    observed_selector = []
                    for label in observed_labels:
                        a_file = os.path.join(fmri_dir, "{}.mat".format(label))
                        if os.path.exists(a_file):
                            encoded_activity_list.append(hdf5storage.loadmat(a_file)["feat"])
                            encoded_labels.append(label)
                            observed_selector.append(True)
                        else:
                            observed_selector.append(False)
                    encoded_activity = np.vstack(encoded_activity_list)
                    encoded_labels = np.array(encoded_labels)
                    observed_selector = np.array(observed_selector)

                    # # Train y mean/norm
                    # train_y_mean = hdf5storage.loadmat(os.path.join(model_dir, 'y_mean.mat'))['y_mean']
                    # train_y_std = hdf5storage.loadmat(os.path.join(model_dir, 'y_norm.mat'))['y_norm']

                    for eval_roi in _rois:
                        print("Evaluation ROI: {}".format(eval_roi))

                        _, target_roi_selector = bdata.select(rois[eval_roi], return_index=True)

                        # If eval_roi is outside train_roi, an error occurs.
                        if np.sum(target_roi_selector[~train_roi_selector]) > 0:
                            raise RuntimeError("Evaluation target ROI is over than training ROI.")

                        # Extract evalution target ROI's observed/encoded brain activity
                        roi_selector = target_roi_selector[train_roi_selector]
                        # y_mean = train_y_mean[:, roi_selector]
                        # y_std = train_y_std[:, roi_selector]
                        pred = encoded_activity[:, roi_selector]
                        obs = observed_activity[:, roi_selector]

                        # Extract fold target observed brain activity
                        obs = obs[observed_selector]
                        # obs_labels = observed_labels[observed_selector]

                        # Evaluation ---------------------------
                        # Profile correlation
                        if not results_db.exists(layer=feature, subject=subject, roi=eval_roi,
                                                 fold=fold, metric='profile_correlation'):
                            results_db.set(layer=feature, subject=subject, roi=eval_roi,
                                           fold=fold, metric='profile_correlation', value=np.array([]))
                            r_prof = profile_correlation(pred, obs)
                            results_db.set(layer=feature, subject=subject, roi=eval_roi,
                                           fold=fold, metric='profile_correlation', value=r_prof)
                            print('Mean profile correlation:     {}'.format(np.nanmean(r_prof)))

                        # # Pattern correlation
                        # if not results_db.exists(layer=feature, subject=subject, roi=eval_roi,
                        #                          fold=fold, metric='pattern_correlation'):
                        #     results_db.set(layer=feature, subject=subject, roi=eval_roi,
                        #                    fold=fold, metric='pattern_correlation',
                        #                    value=np.array([]))
                        #     r_patt = pattern_correlation(pred, obs, mean=y_mean, std=y_std)
                        #     results_db.set(layer=feature, subject=subject, roi=eval_roi,
                        #                    fold=fold, metric='pattern_correlation',
                        #                    value=r_patt)
                        #     print('Mean pattern correlation:     {}'.format(np.nanmean(r_patt)))

                        # # Pair-wise identification accuracy (observed to predicted)
                        # if not results_db.exists(layer=feature, subject=subject, roi=eval_roi,
                        #                          fold=fold, metric='identification_accuracy'):
                        #     results_db.set(layer=feature, subject=subject, roi=eval_roi,
                        #                    fold=fold, metric='identification_accuracy',
                        #                    value=np.array([]))
                        #     if average_sample:
                        #         ident = pairwise_identification(obs, pred)
                        #     else:
                        #         ident = pairwise_identification(obs, pred, single_trial=True,
                        #                                         pred_labels=obs_labels, true_labels=encoded_labels)
                        #     results_db.set(layer=feature, subject=subject, roi=eval_roi,
                        #                    fold=fold, metric='identification_accuracy', value=ident)
                        #     print('Mean identification accuracy: {}'.format(np.nanmean(ident)))

                        # # Pair-wise identification accuracy (predicted to observed)
                        # if not results_db.exists(layer=feature, subject=subject, roi=eval_roi,
                        #                          fold=fold, metric='identification_accuracy_predictedbase'):
                        #     results_db.set(layer=feature, subject=subject, roi=eval_roi,
                        #                    fold=fold, metric='identification_accuracy_predictedbase',
                        #                    value=np.array([]))
                        #     if average_sample:
                        #         ident = pairwise_identification(pred, obs)
                        #     else:
                        #         ident = pairwise_identification(pred, obs, single_trial=True,
                        #                                         pred_labels=encoded_labels, true_labels=obs_labels)
                        #     results_db.set(layer=feature, subject=subject, roi=eval_roi,
                        #                    fold=fold, metric='identification_accuracy_predictedbase',
                        #                    value=ident)
                        #     print('Mean identification accuracy (predicted base): {}'.format(np.nanmean(ident)))

    # Pooled accuracy
    if os.path.exists(output_file_pooled):
        print('Loading {}'.format(output_file_pooled))
        pooled_db = ResultsStore(output_file_pooled)
    else:
        print('Creating new evaluation result store')
        keys = ["layer", "subject", "roi", "metric"]
        pooled_db = ResultsStore(output_file_pooled, keys=keys)

    done_all = True  # Flag indicating that all conditions have been pooled
    for layer, subject, roi, metric in product(features, list(fmri.keys()), rois, metrics):
        # Check if pooling is done
        if pooled_db.exists(layer=layer, subject=subject, roi=roi, metric=metric):
            continue
        pooled_db.set(layer=layer, subject=subject, roi=roi, metric=metric, value=np.array([]))

        # Check if all folds are complete
        done = True
        for fold in cv_labels:
            if not results_db.exists(layer=layer, subject=subject, roi=roi,
                                     fold=fold, metric=metric):
                done = False
                break

        # When all folds are complete, pool the results.
        if done:
            acc = []
            for fold in cv_labels:
                acc.append(results_db.get(layer=layer, subject=subject, roi=roi,
                                          fold=fold, metric=metric))
            if pooled_operation[metric] == "mean":
                acc = np.nanmean(acc, axis=0)
            elif pooled_operation[metric] == "concat":
                acc = np.hstack(acc)
            pooled_db.set(layer=layer, subject=subject, roi=roi,
                          metric=metric, value=acc)

        # If there are any unfinished conditions,
        # do not pool the results and set the done_all flag to False.
        else:
            pooled_db.delete(layer=layer, subject=subject, roi=roi, metric=metric)
            done_all = False
            continue

    if done_all:
        print('All pooling done.')
    else:
        print("Some pooling has not finished.")

    return output_file_pooled, output_file_fold


# Run
if __name__ == "__main__":

    cfg = init_hydra_cfg()

    analysis_name = cfg["_run_"]["name"] + '-' + cfg["_run_"]["config_name"]

    fmri = {
        subject["name"]: subject["paths"]
        for subject in cfg["encoder"]["fmri"]["subjects"]
    }
    training_rois = {
        roi["name"]: roi["select"]
        for roi in cfg["encoder"]["fmri"]["rois"]
    }
    rois = {
        roi["name"]: roi["select"]
        for roi in cfg["evaluation"]["fmri"]["rois"]
    }
    features = [layer["name"] for layer in cfg["encoded_fmri"]["features"]["layers"]]
    label_key = cfg["encoded_fmri"]["fmri"]["label_key"]

    encoder_dir = cfg["encoder"]["path"]
    encoded_fmri_dir = cfg["encoded_fmri"]["path"]

    average_sample = cfg.encoded_fmri.parameters.get("average_sample", True)

    cv_folds = cfg.cv.get("folds", None)

    cv_evaluate_predicted_fmri(
        encoded_fmri_dir,
        encoder_dir,
        fmri,
        training_rois,
        rois,
        features,
        cv_folds,
        label_key,
        output_file_pooled=os.path.join(encoded_fmri_dir, 'evaluation.db'),
        output_file_fold=os.path.join(encoded_fmri_dir, 'evaluation_fold.db'),
        average_sample=average_sample,
    )
