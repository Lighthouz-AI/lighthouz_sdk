from setuptools import find_packages, setup

setup(
    # Basic info
    name="lighthouz",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/my_sdk",
    description="A brief description of your SDK",
    # long_description=open("README.md").read(),
    long_description_content_type="text/markdown",  # This is important for a Markdown README file
    # Choose a license
    license="MIT",
    # Find all packages in the project
    packages=find_packages(),
    # Include package data (like non-python files, e.g., .txt, .md)
    include_package_data=True,
    # Dependencies
    install_requires=["requests", "marshmallow"],
    # Python version requirement
    python_requires=">=3.8",  # Your SDK's Python version requirement
    # Classifiers help users find your project
    classifiers=[
        "Development Status :: 4 - Beta",  # Change as appropriate
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    # Entry points can be used to create executable commands
    entry_points={
        "console_scripts": [
            # Format: 'command=module:function'
            # e.g., 'mycommand=my_sdk.cli:main',
        ],
    },
    # Custom meta-data
    keywords="sdk api development",
    # project_urls={
    #     'Bug Reports': 'https://github.com/yourusername/my_sdk/issues',
    #     'Source': 'https://github.com/yourusername/my_sdk',
    # },
)
