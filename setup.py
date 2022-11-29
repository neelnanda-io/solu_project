from setuptools import setup

setup(
    name="solu",
    version="1.0.0",
    license="LICENSE",
    description="Code to train and analyse SoLU models",
    install_requires=[
        "einops",
        "numpy",
        "torch",
        "datasets",
        "transformers",
        "tqdm",
        "pandas",
        "datasets",
        "wandb",
        "fancy_einsum",
    ],
    packages=["solu"],
)
