from setuptools import setup, find_packages

setup(
    name="peag",
    version="2.0.0",
    description="Patient-context Enhanced Longitudinal Multimodal Alignment and Generation Framework",
    author="PEAG Team",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "tqdm>=4.60.0"
    ],
    python_requires=">=3.8",
)
