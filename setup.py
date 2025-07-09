from setuptools import setup, find_packages

setup(
    name='Part_X',
    version='1.0.0',
    author='Yumeng Cao',
    author_email='ycao108@asu.edu',  # 可选：替换为你自己的邮箱
    description='A level-set classification algorithm using Gaussian Process modeling and partitioning',
    long_description=open('README_LevelSet_Classifier.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yumengcao/Part-X/blob/score-based-algo/',  # 可选：填写你托管代码的链接
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'matplotlib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    include_package_data=True,
    package_data={
        '': ['*.md', '*.txt'],
    },
    entry_points={
        'console_scripts': [
            # 可选：例如 'run-partx=Part_X.part_x:main'
        ]
    },
)