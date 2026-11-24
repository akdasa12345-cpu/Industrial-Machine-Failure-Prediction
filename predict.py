from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts" / "best_machine_failure_model.joblib"
if not MODEL.exists():
    raise SystemExit("Model not found. Run: python src/train.py")

# Example row follows the common AI4I feature names. Adjust if using a modified dataset.
sample = pd.DataFrame([{
    "Type": "M",
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 100
}])
model = joblib.load(MODEL)
prediction = int(model.predict(sample)[0])
print("Machine Failure:", "Yes" if prediction == 1 else "No")
if hasattr(model, "predict_proba"):
    print(f"Estimated failure probability: {model.predict_proba(sample)[0][1]:.3f}")
