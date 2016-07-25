from setuptools import setup

setup(name='beccatest',
version='0.8.1',
description='A test suite for BECCA',
url='http://github.com/brohrer/beccatest',
download_url='https://github.com/brohrer/beccatest/archive/master.zip',
author='Brandon Rohrer',
author_email='brohrer@gmail.com',
license='MIT',
packages=['beccatest'],
install_requires=['becca'],
zip_safe=False)
