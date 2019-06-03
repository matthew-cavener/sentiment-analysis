from setuptools import setup, find_packages

def read_requirements(filename):
    with open(filename, 'r') as f:
        return [
            line.strip() for line in f.readlines() if not(
                line.startswith('#') or line.startswith('--')
            )
        ]

setup(
    name='sentiment',
    version='0.1.0',
    packages=find_packages(),
    zip_safe=False,
    install_requires=read_requirements('requirements.txt')
)
