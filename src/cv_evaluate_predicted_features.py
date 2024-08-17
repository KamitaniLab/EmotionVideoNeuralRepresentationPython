'''Feature decoding (corss-validation) evaluation.'''


from typing import Dict, List, Optional

from itertools import product
import os
import re
import glob

import sys
sys.path.insert(0, "./bdpy/")
import bdpy
print(bdpy.__file__)

from bdpy.dataform import Features, DecodedFeatures, SQLite3KeyValueStore
from bdpy.evals.metrics import profile_correlation, pattern_correlation, pairwise_identification
from bdpy.pipeline.config import init_hydra_cfg
import hdf5storage
import numpy as np
import yaml


# Main #######################################################################

class ResultsStore(SQLite3KeyValueStore):
    """Results store for feature decoding evaluation."""
    pass


def cv_evaluate_predicted_features(
        decoded_feature_path: str,
        true_feature_path: str,
        output_file_pooled: str = './evaluation.db',
        output_file_fold: str = './evaluation_fold.db',
        subjects: Optional[List[str]] = None,
        rois: Optional[List[str]] = None,
        layers: Optional[List[str]] = None,
        cv_folds: Optional[List[Dict]] = None,
        feature_index_file: Optional[str] = None,
        feature_decoder_path: Optional[str] = None,
        average_sample: bool = True,
):
    '''Evaluation of feature decoding.

    Input:

    - deocded_feature_dir
    - true_feature_dir

    Output:

    - output_file

    Parameters:

    TBA
    '''

    # Display information
    print('Subjects: {}'.format(subjects))
    print('ROIs:     {}'.format(rois))
    print('')
    print('Decoded features: {}'.format(decoded_feature_path))
    print('')
    print('True features (Test): {}'.format(true_feature_path))
    print('')
    print('Layers: {}'.format(layers))
    print('')
    if feature_index_file is not None:
        print('Feature index: {}'.format(feature_index_file))
        print('')

    # Loading data ###########################################################
    # True features
    if feature_index_file is not None:
        features_test = Features(true_feature_path, feature_index=feature_index_file)
    else:
        features_test = Features(true_feature_path)

    # get fold name
    if 'name' in cv_folds[0]:
        cv_labels = ['cv-{}'.format(cv['name']) for cv in cv_folds]
    else:
        cv_labels = ['cv-fold{}'.format(icv + 1) for icv in range(len(cv_folds))]

    # Metrics ################################################################
    metrics = ['profile_correlation']  # , 'pattern_correlation', 'identification_accuracy']
    pooled_operation = {
        "profile_correlation": "mean",
        # "pattern_correlation": "concat",
        # "identification_accuracy": "concat",
    }

    # Evaluating decoding performances #######################################

    if os.path.exists(output_file_fold):
        print('Loading {}'.format(output_file_fold))
        results_db = ResultsStore(output_file_fold, timeout=240)
    else:
        print('Creating new evaluation result store')
        keys = ["layer", "subject", "roi", "fold", "metric"]
        results_db = ResultsStore(output_file_fold, timeout=240, keys=keys)

    true_labels = features_test.labels

    for layer in np.random.permutation(layers):
        print('Layer: {}'.format(layer))
        true_y = features_test.get_features(layer=layer)

        for subject, roi, fold in np.random.permutation(list(product(subjects, rois, cv_labels))):
            print('Subject: {} - ROI: {} - Fold: {}'.format(subject, roi, fold))

            # Check if the evaluation is already done
            exists = True
            for metric in metrics:
                exists = exists and results_db.exists(layer=layer, subject=subject, roi=roi, fold=fold, metric=metric)
            if exists:
                print('Already done. Skipped.')
                continue

            # get predicted features
            pred_files = glob.glob(os.path.join(decoded_feature_path, layer, subject, roi, fold, 'decoded_features', '*.mat'))
            if len(pred_files) == 0:
                raise RuntimeError("No decoded features:", os.path.join(decoded_feature_path, layer, subject, roi, fold, '*.mat'))
            pred_y_list = []
            pred_labels = []
            for a_file in pred_files:
                pred_y_list.append(hdf5storage.loadmat(a_file)["feat"])
                pred_label = os.path.splitext(os.path.basename(a_file))[0]
                pred_labels.append(pred_label)
            pred_y = np.vstack(pred_y_list)

            if not average_sample:
                pred_labels = [re.match('trial_\d*-(.*)', x).group(1) for x in pred_labels]

            if not np.array_equal(pred_labels, true_labels):
                y_index = [np.where(np.array(true_labels) == x)[0][0] for x in pred_labels]
                true_y_sorted = true_y[y_index]
            else:
                true_y_sorted = true_y

            # Evaluation ---------------------------

            # Profile correlation
            if results_db.lock(layer=layer, subject=subject, roi=roi, fold=fold, metric='profile_correlation'):
                r_prof = profile_correlation(pred_y, true_y_sorted)
                results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='profile_correlation',
                               value=r_prof)
                print('Mean profile correlation:     {}'.format(np.nanmean(r_prof)))
            else:
                print('Already locked. Skipped.')

    print('All fold done')

    # Pooled accuracy
    if os.path.exists(output_file_pooled):
        print('Loading {}'.format(output_file_pooled))
        pooled_db = ResultsStore(output_file_pooled)
    else:
        print('Creating new evaluation result store')
        keys = ["layer", "subject", "roi", "metric"]
        pooled_db = ResultsStore(output_file_pooled, keys=keys)

    done_all = True  # Flag indicating that all conditions have been pooled
    for layer, subject, roi, metric in product(layers, subjects, rois, metrics):
        # Check if pooling is done
        if pooled_db.exists(layer=layer, subject=subject, roi=roi, metric=metric):
            continue
        pooled_db.set(layer=layer, subject=subject, roi=roi, metric=metric, value=np.array([]))

        # Check if all folds are complete
        done = True
        for fold in cv_labels:
            if not results_db.exists(layer=layer, subject=subject, roi=roi, fold=fold, metric=metric):
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


# Entry point ################################################################

if __name__ == '__main__':

    cfg = init_hydra_cfg()

    decoded_feature_path = cfg["decoded_feature"]["path"]
    gt_feature_path      = cfg["decoded_feature"]["features"]["paths"][0]  # FIXME

    feature_decoder_path = cfg["decoded_feature"]["decoder"]["path"]
    subjects = [s["name"] for s in cfg["decoded_feature"]["fmri"]["subjects"]]
    rois = [r["name"] for r in cfg["decoded_feature"]["fmri"]["rois"]]
    layers = cfg["decoded_feature"]["features"]["layers"]
    cv_folds = cfg.cv.get("folds", None)

    feature_index_file = cfg.decoder.features.get("index_file", None)
    average_sample = cfg["decoded_feature"]["parameters"]["average_sample"]

    cv_evaluate_predicted_features(
        decoded_feature_path,
        gt_feature_path,
        output_file_pooled=os.path.join(decoded_feature_path, 'evaluation.db'),
        output_file_fold=os.path.join(decoded_feature_path, 'evaluation_fold.db'),
        subjects=subjects,
        rois=rois,
        layers=layers,
        cv_folds=cv_folds,
        feature_index_file=feature_index_file,
        feature_decoder_path=feature_decoder_path,
        average_sample=average_sample,
    )
