from setuptools import setup, find_packages

setup(
    name="automl-clean",
    version="0.1.0",
    description="Autonomous Data Quality Monitoring for ML Pipelines",
    author="Rahil S",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "jenga-data>=0.0.1",
        "cleanlab>=2.0.0",
    ],
    python_requires=">=3.8",
)
