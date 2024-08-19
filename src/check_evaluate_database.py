# coding: utf-8

"""
DNN Feature decoding/encoding (corss-validation) - cleanup evaluation database script.

This checks whether all evaluation calculations have been completed.
Also, if an evaluation calculation is terminated midway and blank columns remain, these will be deleted.
"""

from typing import Dict, List, Optional
from itertools import product
import os
import sys

from bdpy.pipeline.config import init_hydra_cfg
from bdpy.dataform import SQLite3KeyValueStore


# Main #######################################################################


class ResultsStore(SQLite3KeyValueStore):
    """Results store for feature decoding evaluation."""
    pass


def check_evaluate_database(
    predicted_result_dir: str,
    analysis_type: str,
    subjects: List[str],
    rois: List[str],
    features: List[str],
    cv_folds: Optional[List[Dict[str, List]]],
    output_file_pooled: str = './evaluation.db',
    output_file_fold: str = './evaluation_fold.db',
):
    if 'name' in cv_folds[0]:
        cv_labels = ['cv-{}'.format(cv['name']) for cv in cv_folds]
    else:
        cv_labels = ['cv-fold{}'.format(icv + 1) for icv in range(len(cv_folds))]

    # Display information
    print('Subjects:        {}'.format(subjects))
    print('Evaluation ROIs: {}'.format(rois))
    print('Layers:          {}'.format(features))
    print('Folds:           {}'.format(cv_labels))
    print('')
    print('Predicted result dir: {}'.format(predicted_result_dir))
    print('')
    print('')

    # Metrics ################################################################
    metrics = ['profile_correlation']

    # Check Fold Evaluation database #######################################
    print("Fold calculation")
    if not os.path.exists(output_file_fold):
        print("Not found fold file:", output_file_fold)
        sys.exit()
    else:
        print('Loading {}'.format(output_file_fold))
        results_db = ResultsStore(output_file_fold)

    finish_fold = True
    for subject, feature, fold, roi, metric in product(subjects, features, cv_labels, rois, metrics):
        if results_db.exists(layer=feature, subject=subject, roi=roi, fold=fold, metric=metric):
            if len(results_db.get(layer=feature, subject=subject, roi=roi, fold=fold, metric=metric)) == 0:
                results_db.delete(layer=feature, subject=subject, roi=roi, fold=fold, metric=metric)
                print("Remove emtpy value: {} - {} - {} - {} - {}".format(subject, feature, fold, roi, metric))
                finish_fold = False
        else:
            finish_fold = False
            break
    if finish_fold:
        print("==========")
        print("All conditions are finished.")
        print("==========")
    else:
        print("==========")
        print("!! There are still some conditions that have not been calculated.")
        print("==========")
    print("")

    # Check Fold Evaluation database #######################################
    print("Pooled calculation")
    if not os.path.exists(output_file_pooled):
        print("Not found pooled file:", output_file_pooled)
        sys.exit()
    else:
        print('Loading {}'.format(output_file_pooled))
        pooled_db = ResultsStore(output_file_pooled)

    finish_fold = True
    for subject, feature, fold, eval_roi, metric in product(subjects, features, cv_labels, rois, metrics):
        if pooled_db.exists(layer=feature, subject=subject, roi=eval_roi, metric=metric):
            if len(pooled_db.get(layer=feature, subject=subject, roi=eval_roi, metric=metric)) == 0:
                pooled_db.delete(layer=feature, subject=subject, roi=eval_roi, metric=metric)
                print("Remove emtpy value: {} - {} - {} - {}".format(subject, feature, eval_roi, metric))
                finish_fold = False
        else:
            finish_fold = False
            break
    if finish_fold:
        print("==========")
        print("All conditions are finished.")
        print("==========")
    else:
        print("==========")
        print("!! There are still some conditions that have not been calculated.")
        print("==========")
    print()

    return output_file_pooled, output_file_fold


# Run
if __name__ == "__main__":

    cfg = init_hydra_cfg()

    analysis_name = cfg["_run_"]["name"] + '-' + cfg["_run_"]["config_name"]

    if "encoder" in cfg:
        analysis_type = "encoding"
        subjects = [subject["name"] for subject in cfg["encoder"]["fmri"]["subjects"]]
        rois = [roi["name"] for roi in cfg["evaluation"]["fmri"]["rois"]]
        features = [layer["name"] for layer in cfg["encoded_fmri"]["features"]["layers"]]
        predicted_result_dir = cfg["encoded_fmri"]["path"]
    elif "decoder" in cfg:
        analysis_type = "decoding"
        subjects = [subject["name"] for subject in cfg["decoded_feature"]["fmri"]["subjects"]]
        rois = [roi["name"] for roi in cfg["decoded_feature"]["fmri"]["rois"]]        
        features = cfg["decoded_feature"]["features"]["layers"]
        predicted_result_dir = cfg["decoded_feature"]["path"]

    cv_folds = cfg.cv.get("folds", None)

    check_evaluate_database(
        predicted_result_dir,
        analysis_type,
        subjects,
        rois,
        features,
        cv_folds,
        output_file_pooled=os.path.join(predicted_result_dir, 'evaluation.db'),
        output_file_fold=os.path.join(predicted_result_dir, 'evaluation_fold.db'),
    )
