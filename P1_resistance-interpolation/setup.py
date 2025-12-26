"""Setup script for Ship Performance Interpolation package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ship-interpolation",
    version="0.1.0",
    author="Applied Mathematics Candidate",
    description="Advanced interpolation methods for ship performance prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nuopel/Navig_P1",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "openpyxl>=3.1.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "pylint>=2.17.0",
            "mypy>=1.4.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
)
