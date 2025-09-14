from setuptools import setup, find_packages

setup(
    name="marketing_mix_model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if not line.startswith("#")
    ],
    author="Your Name",
    description="Marketing Mix Modeling Project",
    python_requires=">=3.8",
)