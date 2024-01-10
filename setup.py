from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    # Basic info
    name="lighthouz",
    version="0.1.0",
    author="Lighthouz AI, Inc",
    author_email="srijan@lighthouz.ai",
    url="https://github.com/Lighthouz-AI/lighthouz_sdk",
    description="Lighthouz AI Python SDK",
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    # Dependencies
    # install_requires=["requests", "marshmallow"],
    install_requires=requirements,
    # Python version requirement
    python_requires=">=3.8",  # Your SDK's Python version requirement
    # Classifiers help users find your project
    classifiers=[
        "Development Status :: 4 - Beta",  # Change as appropriate
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    # Custom meta-data
    keywords="lighthouz python sdk api development",
    project_urls={
        "Bug Reports": "https://github.com/Lighthouz-AI/lighthouz_sdk/issues",
        "Source": "https://github.com/Lighthouz-AI/lighthouz_sdk",
    },
)
