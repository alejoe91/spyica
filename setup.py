import setuptools

pkg_name="spyica"

setuptools.setup(
    name=pkg_name,
    version="0.0.1",
    author="Alessio Buccino",
    author_email="alessiop.buccino@gmail.com",
    description="Spike sorting based on ICA and ORICA",
    url="https://github.com/alejoe91/spyica",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'ipython',
        'spikeinterface',
        'neo',
        'quantities'
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    )
)
