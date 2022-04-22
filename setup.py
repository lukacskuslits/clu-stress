import setuptools

requires = ['pandas>=1.1.5',
            'sklearn>=0.0',
            'scipy>=1.7.3',
            'matplotlib>=3.5.1',
            'xlrd>=2.0.1']

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="clu-stress",
    version="0.0.11",
    author="Lukacs Kuslits",
    author_email="kuslits.lukacs@epss.hu",
    description="A non-parametric clustering algorithm.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lukacskuslits/clu-stress",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
    install_requires=requires,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)

# package_dir={"": "clustress"},
# packages=setuptools.find_packages(where="clustress"),