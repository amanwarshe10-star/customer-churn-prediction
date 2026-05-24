import shap
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import prepare_data

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data
X_train, X_test, y_train, y_test, feature_names = prepare_data()
X_test_df = pd.DataFrame(X_test, columns=feature_names)

# Create SHAP explainer
print("Generating SHAP explanation...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_df)

# Plot - Which features matter most
shap.summary_plot(shap_values, X_test_df, show=False)
plt.title("Feature Importance - SHAP")
plt.tight_layout()
plt.savefig('shap_plot.png', bbox_inches='tight')
plt.show()

print("SHAP plot saved!")
print("Done!")