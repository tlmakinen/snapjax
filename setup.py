
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="snapjax",
    version="0.10",
    author="Lucas Makinen",
    author_email="l.makinen21@imperial.ac.uk",
    description="snapjax: a fast, flexible SuperNova Analysis Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="git@github.com:tlmakinen/snapjax.git",
    packages=["snapjax"],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3",
    install_requires=[
          "jax>0.4.0",
          "numpyro>=0.12.1",
          "jax-cosmo>=0.1.0",
          "tqdm>=4.48.2",
          "numpy>=1.16.0",
          "scipy>=1.4.1",
          "corner",
          "matplotlib"],
)