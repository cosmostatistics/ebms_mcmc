from setuptools import setup, find_packages

#HTTPS_GITHUB_URL = "https://github.com/cosmostatistics/ebms_mcmc

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ["numpy", "scipy", "pyyaml"]

setup(
    name="ebms_mcmc",
    version="1.0.0",
    author="Tobias Röspel, Benedikt Schosser",
    author_email="tobias.roespel@stud.uni-heidelberg.de, schosser@stud.uni-heidelberg.de",
    description="Markov walk exploration of model spaces",
    long_description=long_description,
    long_description_content_type="text/md",
    #url=HTTPS_GITHUB_URL,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    packages=find_packages(),#exclude=["tests"]),
    install_requires=requirements,
    #entry_points={"console_scripts": ["ebms_mcmc=ebms_mcmc.__main__:main"]},
)