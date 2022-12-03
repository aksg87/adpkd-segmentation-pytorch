from setuptools import setup, find_packages
import platform

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

if 'win' in platform.platform().lower() ;
    with open("requirements.windows.addon.txt") as f:
        requirements += f.read().splitlines()

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
