from setuptools import setup, find_packages

setup(
    name="hnet-scene-graph",
    version="0.1.0",
    description="Scene Graph Generation using H-Net Architecture",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.5.1",
        "torchvision>=0.20.1",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "transformers>=4.30.0",
        "einops>=0.7.0",
        "omegaconf>=2.3.0",
        "pandas>=2.0.0",
        "opencv-python>=4.8.0",
        "pycocotools>=2.0.6",
        "h5py>=3.8.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
        "networkx>=3.1",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "flash": [
            "flash-attn>=2.0.0",
        ],
    },
)