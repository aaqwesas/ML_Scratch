from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import joblib
import time
import logging

class RandomForestWithPreprocessing:
    def __init__(self, n_estimators=100, random_state=42, test_size=0.1, pca_variance_ratio=0.95, svd_solver="auto"):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.test_size = test_size
        self.pca_variance_ratio = pca_variance_ratio
        self.svd_solver = svd_solver

        # Logger setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_data(self, X=None, y=None):
        """
        Load the dataset.
        """
        self.logger.info("Splitting dataset into training and testing sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def create_pipeline(self):
        """
        Create a pipeline that includes preprocessing (standardization, PCA) and the Random Forest model.
        """
        self.logger.info("Creating the preprocessing and model pipeline...")
        
        steps = []
        # Add standardization step
        steps.append(("scaler", StandardScaler()))
        
        # Add PCA step if variance ratio is specified
        if self.pca_variance_ratio:
            steps.append(("pca", PCA(n_components=self.pca_variance_ratio, svd_solver=self.svd_solver)))
        
        # Add Random Forest model
        steps.append(("random_forest", RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state
        )))
        
        # Create the pipeline
        self.pipeline = Pipeline(steps)

    def train_model(self):
        """
        Train the pipeline (preprocessing + model) on the training data with a custom progress bar.
        """
        self.logger.info("Training the model pipeline with progress tracking...")
        start_time = time.time()

        # Extract the Random Forest model from the pipeline
        random_forest = self.pipeline.named_steps["random_forest"]

        # Ensure warm_start is enabled for incremental training
        if not random_forest.warm_start:
            random_forest.set_params(warm_start=True)

        # Fit the StandardScaler on the training data
        scaler = self.pipeline.named_steps["scaler"]
        X_train_scaled = scaler.fit_transform(self.X_train)

        # Fit PCA (if enabled) on the scaled training data
        if "pca" in self.pipeline.named_steps:
            pca = self.pipeline.named_steps["pca"]
            X_train_transformed = pca.fit_transform(X_train_scaled)
        else:
            X_train_transformed = X_train_scaled  # Skip PCA if not in the pipeline

        # Manually train the Random Forest incrementally with a progress bar
        for i in tqdm(range(1, self.n_estimators + 1), desc="Training Progress", unit="tree"):
            random_forest.set_params(n_estimators=i)
            random_forest.fit(X_train_transformed, self.y_train)

        # Update the pipeline with the fully trained Random Forest
        self.pipeline.named_steps["random_forest"] = random_forest

        end_time = time.time()
        self.logger.info(f"Training completed in {end_time - start_time:.2f} seconds")

    def evaluate_model(self):
        """
        Evaluate the pipeline on the test data.
        """
        self.logger.info("Evaluating the model pipeline...")
        y_pred = self.pipeline.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.logger.info(f"Test Accuracy: {accuracy:.4f}")
        self.logger.info("\nClassification Report:\n" + classification_report(self.y_test, y_pred))

    def save_pipeline(self, path="model_pipeline.pkl"):
        """
        Save the entire pipeline to a file.
        """
        self.logger.info(f"Saving the pipeline to {path}...")
        joblib.dump(self.pipeline, path)

    def load_pipeline(self, path="model_pipeline.pkl"):
        """
        Load the entire pipeline from a file.
        """
        self.logger.info(f"Loading the pipeline from {path}...")
        self.pipeline = joblib.load(path)

    def run(self, X=None, y=None, pipeline_path="model_pipeline.pkl"):
        """
        Run the entire pipeline: load data, preprocess, attempt to load a saved pipeline first.
        If loading fails, train a new pipeline, save it, and evaluate.

        Parameters:
        - X (array-like): Feature matrix. If None, data must be loaded within the method.
        - y (array-like): Target vector. If None, data must be loaded within the method.
        - pipeline_path (str): Path to save or load the pipeline file.
        """
        self.logger.info("Loading and preprocessing the data...")
        self.load_data(X, y)

        try:
            # Attempt to load the saved pipeline
            self.logger.info("Attempting to load the saved pipeline...")
            self.load_pipeline(pipeline_path)
            self.logger.info("Pipeline loaded successfully.")
        except FileNotFoundError:
            # If the pipeline file does not exist, create and train a new pipeline
            self.logger.warning("No saved pipeline found. Creating and training a new pipeline...")
            self.create_pipeline()
            self.train_model()
            self.save_pipeline(pipeline_path)
            self.logger.info(f"Pipeline saved to {pipeline_path}.")
        
        # Evaluate the pipeline
        self.evaluate_model()



if __name__ == "__main__":
    data = fetch_openml('mnist_784', version=1)
    X, y = data.data, data.target
    rf_model = RandomForestWithPreprocessing(n_estimators=100, random_state=42, pca_variance_ratio=0.5)
    rf_model.run(X, y)