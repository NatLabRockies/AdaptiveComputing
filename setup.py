from setuptools import setup, find_packages

setup(
    name="adaptive_computing",
    version="1.0",
    license="BSD-3-Clause",
    packages=find_packages(),
    package_data={
        "adaptive_computing.hpc.templates": ["*.py", "batch_scripts/*.sh"],
    },
    entry_points={
        "console_scripts": [
            "ac-install-scripts=adaptive_computing.hpc.install_scripts:main",
            "ac-kill-scheduler-jobs=adaptive_computing.hpc.kill_scheduler_jobs:main",
        ],
    },
    install_requires=[
        "smt",
        "scikit-learn==1.8.0",
        "soogo==2.0.1",
        "pydantic",
    ],
    extras_require={
        "agents": [
            "langchain-core",
            "langchain-openai",
            "langgraph",
            "typing_extensions",
            "fastmcp",
        ],
    },
    # Note: tf-melt and Hero are installed separately due to git repository requirements
)
