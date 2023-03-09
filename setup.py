from setuptools import find_packages, setup

setup(
    name="sonn",
    packages=find_packages(),
    version="0.1.0",
    description="Implements convenient ways to train PyTorch models and reduces boilerplate.",
    author="Mert Duman",
    license="MIT",
    install_requires=["torch", "numpy", "matplotlib"]
)
