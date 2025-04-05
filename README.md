To use SMOTE:
Install imblearn on your system:

bash
Copy
Edit
pip install imblearn
Then uncomment:

python
Copy
Edit
from imblearn.over_sampling import SMOTE
...
X_scaled, y = SMOTE(random_state=42).fit_resample(X_scaled, y)
