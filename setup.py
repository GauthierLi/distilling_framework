import os 
from setuptools import setup, find_packages

def get_requirements():
    req_file = r"requirements.txt"
    dependences = []
    with open(req_file) as f:
        lines = f.readlines()
        for line in lines:
            dependences.append(line.strip())
    return dependences

setup(
    name='distill',
    version='0.1.0',
    author='Gauthierli',
    author_email='lwklxh@163.com',
    description='Distilling train framework',
    long_description=open('README.md').read(),
    packages=find_packages(),
    classifiers=[
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.10',
        ],
    install_requires=get_requirements(),
    setup_requires=[
        "torch",
        "torchvision"    
    ],
    # entry_points={
    #     'console_scripts': [
    #         'your-script=your_package.module:main_function',
    #     ],
    # },
    # 更多配置...
)
