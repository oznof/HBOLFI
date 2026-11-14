import re
import io
import os
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


version = find_version("hbolfi", "__init__.py")
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()


setup(
    name='hbolfi',
    version=version,
    packages=['hbolfi', 'hbolfi.settings', 'hbolfi.means', 'hbolfi.utils', 'hbolfi.models',
              'hbolfi.models.gp', 'hbolfi.kernels', 'hbolfi.bayes_opt', 'hbolfi.inference', 'hbolfi.simulators',
              'hbolfi.acquisitions', 'hbolfi.distributions', 'hbolfi.kernel_discovery',
              'hbolfi.kernel_discovery.additive'],
    url='',
    license='MIT',
    author='Afonso Eduardo',
    author_email='afonso.eduardo@ed.ac.uk',
    description='High-dimensional Bayesian Optimization for Likelihood-Free Inference',
    python_requires=">=3.6",
    install_requires=requirements
)
