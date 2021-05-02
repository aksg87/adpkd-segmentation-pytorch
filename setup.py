from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="adpkd-segmentation",
    version="1.0",
    description="ADPKD Segmentation model in PyTorch",
    author="Akshay Goel",
    author_email="akshay.k.goel@gmail.com",
    packages=find_packages(
        include=["adpkd_segmentation", "adpkd_segmentation.*"]
    ),
    license="MIT",
    install_requires=requirements,
    long_description=open("README.md").read(),
)
