from setuptools import setup, find_packages

setup(
    name="introspectai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "accelerate",
        "numpy",
        "scipy",
        "scikit-learn",
        "tqdm",
        "fire",
        "hydra-core",
        "jsonlines",
    ],
)
