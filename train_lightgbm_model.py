import pandas as pd
import lightgbm
import numpy as np
import os
import json
import pickle

# Create directory for saved models
os.makedirs("./saved_models", exist_ok=True)

# Import necessary functions from model.py
from paper_codebase.model import get_proname, combine, norm_score, gettitle, getbody

# Define feature sets (copied from model.py)
xnames_sub_cumu = [
    # General OSS experience
    "clsallcmt",
    "clsallpr",
    "clsalliss",
    "clspronum",
    "clsiss",
    "clsallprreview",
]

xnames_sub_act = [
    # Activeness
    "clsonemonth_cmt",
    "clstwomonth_cmt",
    "clsthreemonth_cmt",
    "clsonemonth_pr",
    "clstwomonth_pr",
    "clsthreemonth_pr",
    "clsonemonth_iss",
    "clstwomonth_iss",
    "clsthreemonth_iss",
]

xnames_sub_sen = [
    # Sentiment
    "clsissuesenmean",
    "clsissuesenmedian",
    "clsprsenmean",
    "clsprsenmedian",
]

xnames_sub_clssolvediss = [  ##Expertise preference
    # Content preference
    "solvedisscos_sim",
    "solvedisscos_mean",
    "solvedissjaccard_sim",
    "solvedissjaccard_sim_mean",
    "solvedissuelabel_sum",
    "solvedissuelabel_ratio",
]

xnames_sub_clsrptiss = [  ##Expertise preference
    # Content preference
    "issjaccard_sim",
    "issjaccard_sim_mean",
    "isscos_sim",
    "isscos_mean",
    "issuelabel_sum",
    "issuelabel_ratio",
]

xnames_sub_clscomtiss = [  ##Expertise preference
    # Content preference
    "commentissuelabel_sum",
    "commentissuelabel_ratio",
    "commentissuecos_sim",
    "commentissuecos_sim_mean",
    "commentissuejaccard_sim",
    "commentissuejaccard_sim_mean",
]

xnames_sub_clscmt = [  ##Expertise preference
    # Content preference
    "cmtjaccard_sim",
    "cmtjaccard_sim_mean",
    "cmtcos_sim",
    "cmtcos_mean",
]

xnames_sub_clspr = [  ##Expertise preference
    # Content preference
    "prjaccard_sim",
    "prjaccard_sim_mean",
    "prcos_sim",
    "prcos_mean",
    "prlabel_sum",
    "prlabel_ratio",
]

xnames_sub_clsprreview = [  ##Expertise preference
    # Content preference
    "prreviewcos_sim",
    "prreviewcos_sim_mean",
    "prreviewjaccard_sim",
    "prreviewjaccard_sim_mean",
]

xnames_sub_clscont = (
    xnames_sub_clscmt
    + xnames_sub_clspr
    + xnames_sub_clsprreview
    + xnames_sub_clsrptiss
    + xnames_sub_clscomtiss
    + xnames_sub_clssolvediss
    + ["lan_sim"]
)

xnames_sub_domain = [  ##Expertise preference
    # Domain preference
    "readmecos_sim_mean",
    "readmecos_sim",
    "readmejaccard_sim_mean",
    "readmejaccard_sim",
    "procos_mean",
    "procos_sim",
    "projaccard_mean",
    "projaccard_sim",
    "prostopic_sum",
    "prostopic_ratio",
]

xnames_sub_isscont = [  ##Candidate issues
    # Content of issues
    "LengthOfTitle",
    "LengthOfDescription",
    "NumOfCode",
    "NumOfUrls",
    "NumOfPics",
    "buglabelnum",
    "featurelabelnum",
    "testlabelnum",
    "buildlabelnum",
    "doclabelnum",
    "codinglabelnum",
    "enhancelabelnum",
    "gfilabelnum",
    "mediumlabelnum",
    "majorlabelnum",
    "triagedlabelnum",
    "untriagedlabelnum",
    "labelnum",
    "issuesen",
    "coleman_liau_index",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "automated_readability_index",
]

xnames_sub_back = [  ##Candidate issues
    # Background of issues
    "pro_gfi_ratio",
    "pro_gfi_num",
    "proclspr",
    "crtclsissnum",
    "pro_star",
    "openiss",
    "openissratio",
    "contributornum",
    "procmt",
    "rptcmt",
    "rptiss",
    "rptpr",
    "rptpronum",
    "rptallcmt",
    "rptalliss",
    "rptallpr",
    "rpt_reviews_num_all",
    "rpt_max_stars_commit",
    "rpt_max_stars_issue",
    "rpt_max_stars_pull",
    "rpt_max_stars_review",
    "rptisnew",
    "rpt_gfi_ratio",
    "ownercmt",
    "owneriss",
    "ownerpr",
    "ownerpronum",
    "ownerallcmt",
    "owneralliss",
    "ownerallpr",
    "owner_reviews_num_all",
    "owner_max_stars_commit",
    "owner_max_stars_issue",
    "owner_max_stars_pull",
    "owner_max_stars_review",
    "owner_gfi_ratio",
    "owner_gfi_num",
]

# Combine all feature sets for the best model
xnames_LambdaMART = (
    xnames_sub_cumu
    + xnames_sub_act
    + xnames_sub_sen
    + xnames_sub_clscont
    + xnames_sub_domain
    + xnames_sub_isscont
    + xnames_sub_back
)


def train_and_save_lightgbm_model():
    """
    Train the LightGBM model on all available data and save it to disk
    """
    print("Loading datasets...")

    # Load issue texts for title and body retrieval
    with open("./data/isstexts.json") as f:
        issuestr = json.load(f)
    issuedata = issuestr["0"]
    lst = []
    for i in range(len(issuedata)):
        lst.append(issuedata[str(i)])
    global dfall
    dfall = pd.DataFrame(lst)

    # Define dataset paths
    datasetname = "simcse"  # Use the best embedding model
    path_name = "./data/dataset_"

    # Load all datasets for training
    print("Loading training datasets...")
    datasetlst = []
    for i in range(19):  # Use all available data for the final model
        datasetlst.append(
            pd.read_pickle(path_name + datasetname + "_" + str(i) + ".pkl")
        )

    # Split into training and validation
    training_set = pd.concat(datasetlst[:-1], axis=0)
    valid_set = datasetlst[-1]  # Use the last fold for validation

    print(f"Training set size: {len(training_set)}")
    print(f"Validation set size: {len(valid_set)}")

    # Train the model
    print("Training LightGBM model...")
    idname = "issgroupid"

    # Prepare training data
    qids_train = training_set.groupby(idname)[idname].count().to_numpy()
    X_train = training_set[xnames_LambdaMART]
    y_train = training_set[["match"]]

    # Prepare validation data
    qids_validation = valid_set.groupby(idname)[idname].count().to_numpy()
    X_validation = valid_set[xnames_LambdaMART]
    y_validation = valid_set[["match"]]

    # Initialize and train the model
    model = lightgbm.LGBMRanker(
        objective="lambdarank",
    )

    model.fit(
        X=X_train,
        y=y_train,
        group=qids_train,
        eval_set=[(X_validation, y_validation)],
        eval_group=[qids_validation],
        eval_at=5,
    )

    # Save the model
    model_path = "./saved_models/pfirec_model.txt"
    model.booster_.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Save feature names for later use
    feature_path = "./saved_models/feature_names.json"
    with open(feature_path, "w") as f:
        json.dump(xnames_LambdaMART, f)
    print(f"Feature names saved to {feature_path}")

    # Save a pickle version of the model as well
    pickle_path = "./saved_models/pfirec_model.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Pickle model saved to {pickle_path}")

    return model


if __name__ == "__main__":
    train_and_save_lightgbm_model()
