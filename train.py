import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
from eda import run_eda
import argparse
import boto3
import io

class OlinkClassifier:
    def __init__(self, data_path, label_col='Label'):
        self.data_path = data_path
        self.label_col = label_col
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.models = {}
        self.results = {}

    def load_data(self):
        print("Loading data...")

        print(f"Reading from S3: {self.data_path}")
        s3 = boto3.client("s3")
        bucket, key = self.data_path.replace("s3://", "").split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        
        ## run EDA
        run_eda(df, target_col="Label", output_dir="plots")

        df.dropna(inplace=True)
        X = df.drop(columns=[self.label_col])

        le = LabelEncoder()
        y = le.fit_transform(df[self.label_col])
        self.label_encoder = le
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Data shape: {df.shape}, Train samples: {len(self.y_train)}")

    def add_model(self, name, model, param_grid):
        self.models[name] = {'model': model, 'params': param_grid}

    def train_and_evaluate(self):
        for name, mp in self.models.items():
            print(f"Training {name}...")
            clf = GridSearchCV(
                mp['model'],
                param_grid=mp['params'],
                scoring='accuracy',
                cv=5,
                n_jobs=-1
            )
            clf.fit(self.X_train, self.y_train)
            preds = clf.predict(self.X_test)
            proba = clf.predict_proba(self.X_test)[:, 1]  # For ROC AUC
            acc = accuracy_score(self.y_test, preds)
            prec = precision_score(self.y_test, preds)
            rec = recall_score(self.y_test, preds)
            roc = roc_auc_score(self.y_test, proba)
            
            self.results[name] = {
            'model': clf.best_estimator_,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'roc_auc': roc,
            'best_params': clf.best_params_
            }

            ## feature importance
            model = clf.best_estimator_
            if hasattr(model, "feature_importances_"):
                fi = pd.Series(model.feature_importances_, index=self.X_train.columns)
            elif hasattr(model, "coef_"):
                fi = pd.Series(model.coef_[0], index=self.X_train.columns)
            else:
                fi = None

            self.results[name]['feature_importance'] = fi

            joblib.dump(model, os.path.join("results",f"{name}_model.pkl"))
            print(f" {name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, ROC AUC: {roc:.4f}")

    def generate_report(self, out_file='model_comparison.csv'):
        print("Saving model comparison report...")
        metrics = ['accuracy', 'precision', 'recall', 'roc_auc']
        rows = []
        for name, res in self.results.items():
            row = {metric.capitalize(): res[metric] for metric in metrics}
            row["Best Params"] = str(res["best_params"])
            rows.append(pd.Series(row, name=name))

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join("results",out_file))
        print(f"Report saved to: {out_file}")

        # Optionally, also save feature importances
        for name, res in self.results.items():
            fi = res.get("feature_importance")
            if fi is not None:
                fi.sort_values(ascending=False).to_csv(os.path.join("results",f"{name}_feature_importance.csv"))
                print(f"Feature importance saved: {name}_feature_importance.csv")
                
    def upload_folder_to_s3(self,local_folder, bucket, s3_prefix):
        s3 = boto3.client("s3")
        for root, _, files in os.walk(local_folder):
            for filename in files:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, local_folder)
                s3_path = f"{s3_prefix}/{relative_path}"
                s3.upload_file(local_path, bucket, s3_path)
                print(f"Uploaded: {s3_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--s3_input', type=str, required=True, help="S3 path to input CSV")
    args = parser.parse_args()
    s3_input = "s3://omics-ml/olink_COVID_19_data_labelled.csv"

    os.makedirs("results", exist_ok=True)

    local_csv = "./olink_COVID_19_data_labelled.csv"

    clf = OlinkClassifier(data_path=args.s3_input)
    clf.load_data()


    
    # Add models
    clf.add_model("LogisticRegression", LogisticRegression(max_iter=1000), {
        'C': [0.01, 0.1, 1.0]
    })
    clf.add_model("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
        'max_depth': [2, 3],
        'n_estimators': [50, 100]
    })

    # Train and evaluate
    clf.train_and_evaluate()

    # Save summary
    clf.generate_report()
    
    clf.upload_folder_to_s3("results", "omics-ml", "results")
    clf.upload_folder_to_s3("plots", "omics-ml", "plots")

