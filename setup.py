from   setuptools               import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mini-plane",
    version="0.1.0",
    author="PlanE Contributors",
    description="Simplified interface for PlanE: Representation Learning over Planar Graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mini-PlanE",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "torch-scatter>=2.1.0",
        "networkx>=2.8",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
        ],
        "preprocessing": [
            # Sage is required for full SPQR preprocessing
            # Note: Install via conda: conda install -c conda-forge sage
            # or from https://www.sagemath.org/download.html
        ],
    },
)
