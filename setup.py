from setuptools import setup, find_packages

setup(
    name="churn_prediction",
    version="0.1.0",
    description="Customer churn prediction with threshold tuning",
    author="Maxim Petrushinskiy",
    packages=find_packages(),
    python_requires=">=3.10",
)