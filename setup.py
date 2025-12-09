"""
Setup script for Footstep Audio Generation Pipeline

Install in development mode:
    pip install -e .

Install normally:
    pip install .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = []
    with open(requirements_file) as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                # Handle git URLs
                if line.startswith('clip @'):
                    requirements.append('clip @ git+https://github.com/openai/CLIP.git')
                else:
                    requirements.append(line)
else:
    requirements = []

setup(
    name="footstep-audio-pipeline",
    version="1.0.0",
    author="Yejin Bang",
    author_email="yejinbang718@gmail.com", 
    description="AI-powered video-to-footstep audio generation pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yejinbang/sfx-project",  # Update this
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "archive"]),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "footstep-pipeline=src.main_pipeline:main",
            "footstep-generate=src.audio_generator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.json"],
    },
)
