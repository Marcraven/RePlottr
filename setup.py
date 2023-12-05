from setuptools import setup, find_packages

with open("requirements.txt", "r") as file:
    lines = file.readlines()

packages = [line.strip() for line in lines]

setup(name="donutplot", packages=find_packages(), install_requires=packages)
