import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tilts",
    version="0.0.1",
    author="codepause",
    author_email="",
    description="Trading instruments for large timeseries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/codepause/tilts",
    project_urls={
        "Bug Tracker": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.9",
)