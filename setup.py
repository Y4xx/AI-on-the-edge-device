"""
Setup script for digit recognition package.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="digit-recognition",
    version="1.0.0",
    author="Yassine OUJAMA",
    description="A deep learning project for recognizing handwritten digits using CNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Y4xx/AI-on-the-edge-device",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.15.0",
        "pyyaml>=6.0",
        "seaborn>=0.12.0",
    ],
    entry_points={
        'console_scripts': [
            'digit-train=train:main',
            'digit-predict=predict:main',
        ],
    },
)
