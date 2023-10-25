from setuptools import find_packages, setup

setup(
    name="auxiliary_a2c",
    packages=[package for package in find_packages() if package.startswith("auxiliary_a2c")],
    include_package_data=True,
    python_requires='>=3.9',
    install_requires=[
        "numpy",
        "stable-baselines3>=2.0",
    ],
    description="An extension of SB3's A2C algorithm with an option to add auxiliary losses",
    author="Sagar Patel",
    url="https://github.com/sagar-pa/auxiliary_sb3",
    author_email="sagar.patel@uci.edu",
    license="MIT",
    version=0.15,
)
