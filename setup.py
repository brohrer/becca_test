from setuptools import setup

setup(
    name='becca_test',
    version='0.8.3',
    description='A test suite for Becca',
    url='http://github.com/brohrer/becca_test',
    download_url='https://github.com/brohrer/becca_test/archive/master.zip',
    author='Brandon Rohrer',
    author_email='brohrer@gmail.com',
    license='MIT',
    packages=['becca_test'],
    include_package_data=True,
    install_requires=['becca'],
    zip_safe=False)
