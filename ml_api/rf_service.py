import logging
import os
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any

from .config import Config

logger = logging.getLogger(__name__)


class RFRiskService:
    """Encapsulates loading artifacts, preprocessing, and prediction for the RF Risk Classifier."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.preprocessor = None
        self.model = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model and preprocessor into memory at startup."""
        artifacts_dir = Path(self.config.artifacts_dir)
        prep_path = artifacts_dir / self.config.rf_preprocessor_filename
        model_path = artifacts_dir / self.config.rf_model_filename

        missing_paths = [str(p) for p in [prep_path, model_path] if not p.exists()]
        if missing_paths:
            raise FileNotFoundError(
                "Missing one or more required artifact files: " + ", ".join(missing_paths)
            )

        logger.info("Loading RF preprocessor from %s", prep_path)
        self.preprocessor = joblib.load(prep_path)

        logger.info("Loading RF model from %s", model_path)
        self.model = joblib.load(model_path)

        logger.info("RF Artifacts loaded successfully")

    def predict(self, features: Dict[str, Any]) -> Tuple[float, int, str]:
        """
        Run RF model inference.
        Returns: (confidence_score, predicted_class, predicted_label)
        """
        # Convert dictionary to DataFrame (1 row)
        try:
            df = pd.DataFrame([features])
        except Exception as exc:
            raise ValueError(f"Features must be a dictionary. Error: {exc}") from exc

        # Ensure we drop any excluded columns if they somehow made it in
        drop_cols = ["Device_ID", "Risk_Class", "Maintenance_Report", "Purchase_Date", "Risk_Class_Label"]
        cols_to_drop = [c for c in drop_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Preprocess
        try:
            X_processed = self.preprocessor.transform(df)
        except Exception as exc:
            raise ValueError(f"Failed to preprocess features. Check if required fields are missing. Error: {exc}") from exc

        # Predict
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)

        pred_class = int(predictions[0])
        confidence = float(probabilities[0].max())
        
        risk_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        pred_label = risk_mapping.get(pred_class, "Unknown")

        return confidence, pred_class, pred_label
