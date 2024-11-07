from setuptools import setup, find_packages

setup(
    name="improved_classify",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "accelerate>=1.1.0",
        "timm==0.4.12",
        'matplotlib',
        'numpy',
        'scikit-learn',
        'ipdb',
        'nvitop',
        "GenerativeRL",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
