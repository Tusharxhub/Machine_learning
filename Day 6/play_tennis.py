
# ! write a decision tree classifier program on the play tennis dataset

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def load_play_tennis(file_path: Path) -> pd.DataFrame:
	"""Load dataset from either Excel or CSV content."""
	try:
		# Attempt as Excel first in case the file is a real workbook.
		return pd.read_excel(file_path)
	except Exception:
		# Fallback because the provided .xls file is CSV text.
		return pd.read_csv(file_path)


def main() -> None:
	dataset_path = Path(__file__).parent / "Deta" / "play_tennis.xls"

	if not dataset_path.exists():
		raise FileNotFoundError(f"Dataset not found: {dataset_path}")

	df = load_play_tennis(dataset_path)
	df.columns = [col.strip().lower() for col in df.columns]

	required_cols = ["day", "outlook", "temp", "humidity", "wind", "play"]
	missing = [col for col in required_cols if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")

	x = df[["outlook", "temp", "humidity", "wind"]]
	y = df["play"]

	x_train, x_test, y_train, y_test = train_test_split(
		x,
		y,
		test_size=0.30,
		random_state=42,
		stratify=y,
	)

	preprocessor = ColumnTransformer(
		transformers=[
			(
				"cat",
				OneHotEncoder(handle_unknown="ignore"),
				["outlook", "temp", "humidity", "wind"],
			)
		]
	)

	model = Pipeline(
		steps=[
			("preprocessor", preprocessor),
			("classifier", DecisionTreeClassifier(criterion="entropy", random_state=42)),
		]
	)

	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)

	print("Play Tennis - Decision Tree Classifier")
	print(f"Dataset shape: {df.shape}")
	print(f"Train size: {len(x_train)} | Test size: {len(x_test)}")
	print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
	print("Classification Report:")
	print(classification_report(y_test, y_pred))

	sample = pd.DataFrame(
		[
			{
				"outlook": "Sunny",
				"temp": "Cool",
				"humidity": "High",
				"wind": "Strong",
			}
		]
	)
	sample_pred = model.predict(sample)[0]
	print(
		"Sample prediction for "
		"(Sunny, Cool, High, Strong): "
		f"{sample_pred}"
	)


if __name__ == "__main__":
	main()



