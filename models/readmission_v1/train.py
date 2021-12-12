# Logistic Regression Classifier
import os
import sys
import argparse
import joblib
import pandas as pd
from pprint import pprint

from azureml.core import Run
from azureml.core.run import Run

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def getRuntimeArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str)
    args = parser.parse_args()
    return args


def main():
    args = getRuntimeArgs()
    run = Run.get_context()

    readmission_data_df = pd.read_csv(
        os.path.join(args.data_path, 'diabetic_data.csv'))
    clf = model_train(readmission_data_df, run)

    # copying to "outputs" directory, automatically uploads it to Azure ML
    output_dir = './outputs/'
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(value=clf, filename=os.path.join(output_dir, 'model.pkl'))


def model_train(ds_df, run):
    ds_df['readmitted'] = pd.Series(
        [0 if val == 'NO' else 1 for val in ds_df['readmitted']])

    ds_df.drop(['encounter_id', 'patient_nbr', 'payer_code',
               'weight', 'medical_specialty'], axis=1, inplace=True)

    # remove rows that have NA in 'race', 'diag_1', 'diag_2', or 'diag_3'
    # remove rows that have invalid values in 'gender'
    ds_df = ds_df[ds_df['race'] != '?']
    ds_df = ds_df[ds_df['diag_1'] != '?']
    ds_df = ds_df[ds_df['diag_2'] != '?']
    ds_df = ds_df[ds_df['diag_3'] != '?']
    ds_df = ds_df[ds_df['gender'] != 'Unknown/Invalid']
    # Recategorize 'age' so that the population is more evenly distributed
    ds_df['age'] = pd.Series(['0-20' if val in ['[0-10)', '[10-20)'] else val
                              for val in ds_df['age']], index=ds_df.index)
    ds_df['age'] = pd.Series(['20-40' if val in ['[20-30)', '[30-40)'] else val
                              for val in ds_df['age']], index=ds_df.index)
    ds_df['age'] = pd.Series(['40-60' if val in ['[40-50)', '[50-60)'] else val
                              for val in ds_df['age']], index=ds_df.index)
    ds_df['age'] = pd.Series(['60-80' if val in ['[60-70)', '[70-80)'] else val
                              for val in ds_df['age']], index=ds_df.index)
    ds_df['age'] = pd.Series(['80-100' if val in ['[80-90)', '[90-100)'] else val
                              for val in ds_df['age']], index=ds_df.index)
    # original 'discharge_disposition_id' contains 28 levels
    # reduce 'discharge_disposition_id' levels into 2 categories
    # discharge_disposition_id = 1 corresponds to 'Discharge Home'
    ds_df['discharge_disposition_id'] = pd.Series(['Home' if val == 1 else 'Other discharge'
                                                   for val in ds_df['discharge_disposition_id']], index=ds_df.index)
    # original 'admission_source_id' contains 25 levels
    # reduce 'admission_source_id' into 3 categories
    ds_df['admission_source_id'] = pd.Series(['Emergency Room' if val == 7 else 'Referral' if val == 1 else 'Other source'
                                              for val in ds_df['admission_source_id']], index=ds_df.index)
    # original 'admission_type_id' contains 8 levels
    # reduce 'admission_type_id' into 2 categories
    ds_df['admission_type_id'] = pd.Series(['Emergency' if val == 1 else 'Other type'
                                            for val in ds_df['admission_type_id']], index=ds_df.index)

    # keep only 'insulin' and remove the other 22 diabetes medications
    ds_df.drop(['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
                'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
                'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
                'metformin-rosiglitazone', 'metformin-pioglitazone'], axis=1, inplace=True)
    ds_df['diag_1'] = pd.Series([1 if val.startswith(
        '250') else 0 for val in ds_df['diag_1']], index=ds_df.index)
    ds_df.drop(['diag_2', 'diag_3'], axis=1, inplace=True)

    y_raw = ds_df['readmitted']
    X_raw = ds_df.drop('readmitted', axis=1)
    pprint(X_raw.loc[1, :].to_dict())
    categorical_features = X_raw.select_dtypes(include=['object']).columns
    numeric_features = X_raw.select_dtypes(include=['int64', 'float']).columns

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value="missing")),
        ('onehotencoder', OneHotEncoder(categories='auto', sparse=False))])

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    feature_engineering_pipeline = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ], remainder="drop")

    # Encode Labels
    le = LabelEncoder()
    encoded_y = le.fit_transform(y_raw)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, encoded_y, test_size=0.20, stratify=encoded_y, random_state=42)

    # Create sklearn pipeline
    lr_clf = Pipeline(steps=[('preprocessor', feature_engineering_pipeline),
                             ('classifier', LogisticRegression(solver="lbfgs"))])
    # Train the model
    lr_clf.fit(X_train, y_train)

    # Capture metrics
    train_acc = lr_clf.score(X_train, y_train)
    test_acc = lr_clf.score(X_test, y_test)
    print("Training accuracy: %.3f" % train_acc)
    print("Test data accuracy: %.3f" % test_acc)

    # Log to Azure ML
    run.log('Train accuracy', train_acc)
    run.log('Test accuracy', test_acc)

    return lr_clf


if __name__ == "__main__":
    main()
