import os

import pandas as pd

from aif360.datasets import StandardDataset


default_mappings = {
    'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
    'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-white'},
                                 {1.0: 'Male', 0.0: 'Female'}]
}

class AdultDataset(StandardDataset):
    """Adult Census Income Dataset.

    See :file:`aif360/data/raw/adult/README.md`.
    """

    def __init__(self, 
                 label_name,
                 favorable_classes,
                 protected_attribute_names,
                 privileged_classes,
                 categorical_features,
                 features_to_keep, 
                 instance_weights_name=None,
                 features_to_drop=[],
                 na_values=['?'], custom_preprocessing=None,
                 metadata=default_mappings,
                 ):
    
        X_train = pd.read_pickle('/home/jl0274/Senior Thesis/X_train_norm.pkl')
        y_train = pd.read_pickle('/home/jl0274/Senior Thesis/y_train.pkl')
        X_val = pd.read_pickle('/home/jl0274/Senior Thesis/X_val_norm.pkl')
        y_val = pd.read_pickle('/home/jl0274/Senior Thesis/y_val.pkl')
        X_test = pd.read_pickle('/home/jl0274/Senior Thesis/X_test_norm.pkl')
        y_test = pd.read_pickle('/home/jl0274/Senior Thesis/y_test.pkl')
        
        y_train = y_train.values.reshape(-1, 1)
        X_train['action_taken'] = y_train
        y_val = y_val.values.reshape(-1, 1)
        X_val['action_taken'] = y_val
        y_test = y_test.values.reshape(-1, 1)
        X_test['action_taken'] = y_test
        
        train = X_train
        val = X_val
        test = X_test

        df = pd.concat([train, val, test], ignore_index=True)
        
        # TEST THING SHOULD DELETE AFTER
        # df = pd.read_pickle('/home/jl0274/Senior Thesis/test_df.pkl')

        super(AdultDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
