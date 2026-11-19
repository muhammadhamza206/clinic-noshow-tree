# clinic-noshow-tree

`clinic_appointments.csv` needs to sit in this folder.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python fit.py
```

`fit.py` prints:

- training and test row counts, and the number of no-shows in each
- the all-show baseline on the test rows: TN, FP, FN, TP, accuracy, precision, and recall
- the depth-3 tree as text, with training weights
- the same test metrics for that tree (`v1`)
- training accuracy and test accuracy
- total value and value per appointment for the baseline and for `v1` (`-60` per false positive and `-120` per false negative)
- for appointments D and E: the leaf, its training show count, its training no-show count, its no-show rate, and the decision (`overbook` or `hold`)

`fit.py` saves `tree_v1.png`, a picture of the fitted tree.
