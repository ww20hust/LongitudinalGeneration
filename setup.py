from setuptools import setup, find_packages

setup(
    name="peag",
    version="0.1.0",
    description="Patient-context Enhanced Longitudinal Multimodal Alignment and Generation Framework",
    author="PEAG Research Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)

