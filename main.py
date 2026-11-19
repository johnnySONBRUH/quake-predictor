import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

model = RandomForestClassifier(
    n_jobs=-1,
    random_state=42
)
data = pd.read_csv("data.csv", index_col=False)
x = data.drop(columns=["alert"])
y = data["alert"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model.fit(x_train, y_train)
pred = model.predict(x_test)
acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred, average="macro", zero_division=0)
rec = recall_score(y_test, pred, average="macro", zero_division=0)
f1 = f1_score(y_test, pred, average="macro", zero_division=0)
cm = confusion_matrix(y_test, pred)
print(f"ACC: {acc}")
print(f"PRECISION: {prec}")
print(f"RECALL : {rec}")
print(f"F1: {f1}")
print("CM: \n", cm)
joblib.dump(model, "predictor.pkl", compress=3)
