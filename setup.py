from setuptools import setup, find_packages

setup(
    name="AgenticLearnPro",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19",
        "gymnasium>=0.29"  # For RL environments
    ],
    author="Stephanie Ewelu",
    author_email="stephanieewelu@gmail.com",
    description="A simple agentic AI package using reinforcement learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/stephanieewelu/AgenticLearnPro",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)