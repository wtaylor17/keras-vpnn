from setuptools import setup

setup(
    name='keras-vpnn',
    version='1.0.0',
    packages=['vpnn', 'vpnn.layers'],
    url='https://github.com/wtaylor17/keras-vpnn',
    license='MIT',
    author='William Taylor',
    author_email='wtaylor@upei.ca',
    description='Implementation of Volume-Preserving Neural Networks in Keras',
    # requires=['keras', 'tensorflow', 'numpy']
)
