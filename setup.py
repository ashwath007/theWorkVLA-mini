"""Setup configuration for the India Egocentric VLA Data Engine package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
long_description = ""
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Read requirements
requirements = []
req_path = Path(__file__).parent / "requirements.txt"
if req_path.exists():
    requirements = [
        line.strip()
        for line in req_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="vla_engine",
    version="0.1.0",
    author="India VLA Team",
    author_email="vla@example.com",
    description="India Egocentric VLA Data Engine - Capture, process and train on egocentric headset data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/india-vla-engine",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "ruff>=0.1.0",
        ],
        "rpi": [
            "smbus2>=0.4.2",
            "RPi.GPIO>=0.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vla-record=capture.cli:app",
            "vla-pipeline=scripts.run_pipeline:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords="vla robotics egocentric data-pipeline lerobot india",
)
