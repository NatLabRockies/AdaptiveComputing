from setuptools import setup, find_packages

setup(
    name="adaptive_computing",
    version="1.0",
    license="BSD-3-Clause",
    packages=find_packages(),
    install_requires=[
        "smt", 
        "scikit-learn==1.8.0",
        "soogo==2.0.1", 
        "pydantic",
        "langchain-core"
    ]
    # Note: tf-melt and Hero are installed separately due to git repository requirements
)
