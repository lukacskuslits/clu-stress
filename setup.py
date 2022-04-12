import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="clustress-package",
    version="0.0.1",
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
    package_dir={"": "clustress"},
    packages=setuptools.find_packages(where="clustress"),
    python_requires=">=3.5",
)