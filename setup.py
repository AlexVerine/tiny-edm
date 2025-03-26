from setuptools import setup, find_packages

setup(
    name='tiny_edm',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Add your project dependencies here 
        "matplotlib",
        "numpy",
        "pandas",
        "scikit-learn",
        "tqdm"
    ],
    entry_points={
        'console_scripts': [
            # Add your command line scripts here
        ],
    },
    author='Alexandre Verine',
    author_email='alexandre.verine@ens.fr',
    description='A minimal, educational implementation of Elucidated Diffusion Models (EDM) in PyTorch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/alexverine/tiny-edm',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)