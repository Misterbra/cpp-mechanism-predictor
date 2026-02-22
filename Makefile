.PHONY: install test train validate analyze app clean all

# Install dependencies
install:
	pip install -r requirements.txt

# Run unit tests
test:
	python tests/test_features.py

# Train the model
train:
	python src/train_mechanism.py

# Run robust validation (nested CV + bootstrap)
validate:
	python src/robust_validation.py

# Run advanced analysis (error patterns, t-SNE, feature importance)
analyze:
	python src/advanced_analysis.py

# Run feature selection analysis
feature-selection:
	python src/feature_selection_analysis.py

# Run proper nested CV (feature selection inside loop)
nested-cv:
	python src/proper_nested_cv.py

# Launch the web application
app:
	streamlit run app.py

# Full reproducibility pipeline
all: install test train validate analyze
	@echo "Full pipeline completed."

# Clean generated results (keeps data and models)
clean:
	rm -rf results/feature_selection_*
	rm -rf results/proper_nested_cv_*
	rm -rf __pycache__ src/__pycache__
