from setuptools import setup, find_packages

# Dependencies of the vendored Meta-World package. The full experiment
# environment (torch, timm, matplotlib, ...) is listed in requirements.txt.
required = [
    'gym==0.15.4',
    'Cython<3',
    'mujoco-py>=2.1,<2.2',
    'numpy>=1.18',
    'scipy',
    'h5py',
    'flatten_dict',
    'opencv-python',
    'tqdm',
]

setup(
    name="doci",
    version="0.0.1",
    packages=find_packages(where=".", include=["metaworld*"]),
    package_dir={"": "."},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=required,
)
