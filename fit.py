"""Split clinic appointments, score an all-show baseline, and fit a depth-3 tree."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

df = pd.read_csv(Path(__file__).resolve().parent / "clinic_appointments.csv")

y = (df["no_show"] == "no-show").astype(int)
X = df[
    [
        "lead_time_days",
        "prior_no_shows",
        "age",
        "distance_miles",
        "sms_reminder",
        "new_patient",
    ]
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"training rows: {len(X_train)}")
print(f"training no-shows: {int(y_train.sum())}")
print(f"test rows: {len(X_test)}")
print(f"test no-shows: {int(y_test.sum())}")


def report(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    print(name)
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")
    print(f"accuracy: {accuracy:.3f}")
    print(f"precision: {precision:.3f}")
    print(f"recall: {recall:.3f}")


def value(tn, fp, fn, tp):
    return -60 * fp - 120 * fn


baseline_pred = [0] * len(y_test)
report("baseline", y_test, baseline_pred)

feature_names = list(X.columns)
v1 = DecisionTreeClassifier(max_depth=3, random_state=42)
v1.fit(X_train, y_train)

print(export_text(v1, feature_names=feature_names, show_weights=True))

fig, ax = plt.subplots(figsize=(14, 7))
plot_tree(
    v1,
    filled=True,
    feature_names=feature_names,
    class_names=["show", "no-show"],
    fontsize=9,
    ax=ax,
)
fig.savefig(Path(__file__).resolve().parent / "tree_v1.png", bbox_inches="tight")
plt.close(fig)

v1_test_pred = v1.predict(X_test)
report("v1", y_test, v1_test_pred)

train_accuracy = accuracy_score(y_train, v1.predict(X_train))
test_accuracy = accuracy_score(y_test, v1_test_pred)
print(f"training accuracy: {train_accuracy:.3f}    test accuracy: {test_accuracy:.3f}")

for name, y_pred in (("baseline", baseline_pred), ("v1", v1_test_pred)):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    total_value = value(tn, fp, fn, tp)
    print(f"{name} total value: {total_value:.2f}")
    print(f"{name} value per appointment: {total_value / 120:.2f}")


def should_overbook(appointment):
    row = pd.DataFrame([appointment], columns=X_train.columns)
    leaf = int(v1.apply(row)[0])
    weights = v1.tree_.value[leaf][0] * v1.tree_.weighted_n_node_samples[leaf]
    counts = dict(zip(v1.classes_, weights))
    shows = counts[0]
    no_shows = counts[1]
    no_show_rate = no_shows / (shows + no_shows)
    print(f"leaf: {leaf}")
    print(f"shows: {int(shows)}")
    print(f"no-shows: {int(no_shows)}")
    print(f"no-show rate: {no_show_rate:.3f}")
    prediction = int(v1.predict(row)[0])
    return "overbook" if prediction == 1 else "hold"


appointment_d = {
    "lead_time_days": 21,
    "prior_no_shows": 1,
    "age": 27,
    "distance_miles": 9.5,
    "sms_reminder": 0,
    "new_patient": 1,
    "appointment_hour": 15,
}
appointment_e = {
    "lead_time_days": 3,
    "prior_no_shows": 0,
    "age": 61,
    "distance_miles": 2.0,
    "sms_reminder": 1,
    "new_patient": 0,
    "appointment_hour": 9,
}

print("D")
print(f"decision: {should_overbook(appointment_d)}")
print("E")
print(f"decision: {should_overbook(appointment_e)}")
