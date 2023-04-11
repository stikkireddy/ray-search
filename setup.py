from setuptools import setup, find_packages


setup(
    name="ray-search",
    author="Sriharsha Tikkireddy",
    author_email="sri.tikkireddy@databricks.com",
    description="Ray Search",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    packages=find_packages(exclude=['tests', 'tests.*', ]),
    use_scm_version={
        "local_scheme": "dirty-tag"
    },
    setup_requires=['setuptools_scm'],
    install_requires=[
        'ray[default]>=2.3.1',
        'pandas>=1.3',
        'scikit-learn>=1.2.2',
        'torch>=2.0.0',
        'scipy>=1.10.1',
        'cloudpickle>=2.2.1',
        'chromadb>=0.3.21',
        'setuptools>=67',
    ],
    extras_require={
        'cpu': ['faiss-cpu>=1.7.3'],
        'gpu': ['faiss-gpu>=1.7.3']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
