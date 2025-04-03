from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
import pandas as pd
import numpy as np


def load_preproc_data_adult(protected_attributes=None, sub_samp=False, balance=False):
    def custom_preprocessing(df):
        """The custom pre-processing function is adapted from
            https://github.com/fair-preprocessing/nips2017/blob/master/Adult/code/Generate_Adult_Data.ipynb
            If sub_samp != False, then return smaller version of dataset truncated to tiny_test data points.
        """
        df.rename(columns={'applicant_sex': 'sex'}, inplace=True)
        return df

    D_features = ['sex'] if protected_attributes is None else protected_attributes
    Y_features = ['action_taken']
    X_features = ['hoepa_status_2.0', 'hoepa_status_3.0', 'interest_rate',
       'total_loan_costs', 'rate_spread', 'origination_charges',
       'initially_payable_to_institution', 'lender_credits', 'discount_points',
       'negative_amortization', 'interest_only_payment', 'balloon_payment',
       'aus_1', 'manufactured_home_land_property_interest',
       'applicant_race_1_5.0', 'submission_of_application', 'total_units',
       'purchaser_type_1.0', 'reverse_mortgage_2.0',
       'other_nonamortizing_features', 'applicant_sex']
    categorical_features = []

    # privileged classes
    all_privileged_classes = {"sex": [0.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {0.0: 'Male', 1.0: 'Female'}}

    return AdultDataset(
        label_name=Y_features[0],
        # favorable classes means the favorable outcome of the target variable
        favorable_classes=[1.0],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        na_values=['?'],
        metadata={'label_maps': [{1.0: 1.0, 0.0: 0.0}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing)
