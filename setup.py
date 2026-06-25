#!/usr/bin/env python3
"""Setup script for WALDO package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="waldo-anomaly",
    version="1.0.0",
    description="WALDO: Wasserstein-Aligned Localisation via Differential Observations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="WALDO Team",
    author_email="bernhard.kainz@fau.de",
    url="https://github.com/bkainz/WALDO_MICCAI26_demo",
    packages=find_packages(where="."),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "Pillow>=9.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.44.0",  # DINOv3-ViT-B/16 backbone (DINOv2 fallback on older versions)
        "openai>=1.0.0",
        "huggingface_hub>=0.19.0",
        "datasets>=2.14.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "cxr": [
            "pydicom>=2.4.0",
        ],
    },
    # The scripts/ directory is not packaged (it relies on a sys.path shim and is
    # invoked as `python scripts/<name>.py ...`), so no console_scripts entry points.
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="medical-imaging anomaly-detection vision-language zero-shot",
)
