import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RCSnail-Commons",
    version="1.0.6",
    author="Martin Liivak",
    author_email="martin.liivak@gmail.com",
    description="RCSnail AI project's common methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "pyzmq",
        "ruamel.yaml"
    ]
)
