# coding: utf-8

"""
DNN Feature decoding/encoding (corss-validation) - wash training and prediction results script.

This deletes intermediate files that were created when training or prediction was
terminated midway due to an error or interrupt.
If intermediate files remain, re-running the training/prediction script will not calculate that case.

If you run this script and find intermediate files,
please delete them with this script and then run training/prediction again.

Note that if there is a training/prediction process currently running, that execution will also be discarded,
so run this script when all training/prediction processes have finished.
"""

from typing import Dict, List, Optional
from itertools import product
import os
import sys
import sqlite3
import glob
import numpy as np
import shutil
from tqdm import tqdm
from bdpy.pipeline.config import init_hydra_cfg


# Main #######################################################################

def check_training_prediction(
    predicted_result_dir: str,
    target: str,
    analysis_type: str,
    analysis_name: str,
    subjects: List[str],
    rois: List[str],
    features: List[str],
    cv_folds: Optional[List[Dict[str, List]]],
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
    print('Predicted result dir: {}'.format(predicted_result_dir))
    print('')
    print('')

    # Check tmp database for train and predict ########################
    distcomp_db = os.path.join('./tmp', analysis_name + '.db')
    if not os.path.exists(distcomp_db):
        print("Not found tmporal file:", distcomp_db)
        print("Please run it in the same current directory as when you ran train/predict.")
        return
    else:
        running_process_list = []
        with sqlite3.connect(distcomp_db) as conn:
            c = conn.cursor()
            c.execute("select name from computation;")
            for x in c.fetchall():
                running_process_list.append(x[0])
        if len(running_process_list) == 0:
            print("==========")
            print("No processes were terminated during execution.")
            print("==========")
            print()
        else:
            print("The following conditions have not been finished. Please re-run training/prediction script.")
            print("Delete these execution information")
            for running_process in running_process_list:
                print(running_process.replace(analysis_name, ""))

            # Need to remove the process information from tmp database file
            with sqlite3.connect(distcomp_db,  isolation_level='EXCLUSIVE') as conn:
                c = conn.cursor()
                for running_process in running_process_list:
                    c.execute('DELETE FROM computation WHERE name = "%s"' % running_process)
                c.execute("select name from computation;")
            print("Done.")
            print()

    # Check decoded features files ####################################
    if target == "predict":
        print("Check the contents of the predicted dir for all conditions.")
        predicted_filenum_list = {}
        total_num = len(subjects) * len(rois) * len(features) * len(cv_labels)
        for subject, roi, feature, fold in tqdm(product(subjects, rois, features, cv_labels), total=total_num):
            if analysis_type == "decoding":
                search_dir = os.path.join(predicted_result_dir, feature, subject, roi, fold, 'decoded_features')
            else:
                search_dir = os.path.join(predicted_result_dir, feature, subject, roi, fold, 'encoded_fmri')
            pred_files = glob.glob(os.path.join(search_dir, '*.mat'))
            predicted_filenum_list[search_dir] = len(pred_files)
        print("")

        # file数が一定であることを確認
        filenum_list = [fnum for fnum in predicted_filenum_list.values()]
        unique_filenum_list = np.unique(filenum_list)
        if len(unique_filenum_list) == 1 and unique_filenum_list[0] != 0:
            print("==========")
            print("All predicted files exist.")
            print("==========")
        else:
            if False:
                # The case with the most occurrences in terms of number of files is assumed to be the default file num,
                # and a different number of files is picked up.
                # In emotion project, this cannot be used because the number of files differs for each fold.
                most_frequent_filenum = 0
                most_frequent_casenum = 0
                for unum in unique_filenum_list:
                    if most_frequent_casenum < filenum_list.count(unum):
                        most_frequent_casenum = filenum_list.count(unum)
                        most_frequent_filenum = unum
                print("Default number of predicted file:", most_frequent_filenum)
                delete_candidate_list = []
                for decdir, fnum in predicted_filenum_list.items():
                    if fnum != most_frequent_filenum:
                        print("Wrong predicted file num {}: {}".format(fnum, decdir))
                        delete_candidate_list.append(decdir)
                print('')
            else:
                # Detect only directories with 0 files
                delete_candidate_list = []
                for decdir, fnum in predicted_filenum_list.items():
                    if fnum == 0:
                        print("Wrong predicted file num {}: {}".format(fnum, decdir))
                        delete_candidate_list.append(decdir)
                print('')

            if len(delete_candidate_list) != 0:
                # Answer on command line whether to delete the directory
                print("Do you want to delete these unfinished directories?")
                print("Basically, you need to delete it and re-execute it.")
                print("Please check that the default number of files and deletion directory candidates are correct.")
                while True:
                    print("Please respond [y/n]:")
                    res = input()
                    if res == 'y':
                        print("Remove the above directories.")
                        for delete_candidate in delete_candidate_list:
                            shutil.rmtree(delete_candidate)
                        print("Done.")
                        break
                    elif res == 'n':
                        break
            else:
                print("==========")
                print("All predicted files exist.")
                print("==========")

    return


# Run
if __name__ == "__main__":

    cfg = init_hydra_cfg()

    if ("wash" in cfg) and (cfg["wash"] not in ["train", "predict"]):
        raise RuntimeError("Please specify the wash target: '--override +wash=train' or '--override +wash=predict'")
    target = cfg["wash"]

    if "encoder" in cfg:
        analysis_type = "encoding"
        subjects = [subject["name"] for subject in cfg["encoder"]["fmri"]["subjects"]]
        rois = [roi["name"] for roi in cfg["evaluation"]["fmri"]["rois"]]
        features = [layer["name"] for layer in cfg["encoded_fmri"]["features"]["layers"]]
        predicted_result_dir = cfg["encoded_fmri"]["path"]
        if target == "train":
            analysis_sciprt_name = "cv_train_encoder_fastl2lir"
        elif target == "predict":
            analysis_sciprt_name = "cv_predict_fmri_fastl2lir"

    elif "decoder" in cfg:
        analysis_type = "decoding"
        subjects = [subject["name"] for subject in cfg["decoded_feature"]["fmri"]["subjects"]]
        rois = [roi["name"] for roi in cfg["decoded_feature"]["fmri"]["rois"]]
        features = cfg["decoded_feature"]["features"]["layers"]
        predicted_result_dir = cfg["decoded_feature"]["path"]
        if target == "train":
            analysis_sciprt_name = "cv_train_decoder_fastl2lir"
        elif target == "predict":
            analysis_sciprt_name = "cv_predict_feature_fastl2lir"

    cv_folds = cfg.cv.get("folds", None)
    analysis_name = analysis_sciprt_name + "-" + cfg["_run_"]["config_name"]

    check_training_prediction(
        predicted_result_dir,
        target,
        analysis_type,
        analysis_name,
        subjects,
        rois,
        features,
        cv_folds,
    )
