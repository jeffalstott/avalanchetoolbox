from distutils.core import setup
setup(
        name = 'avalanchetoolbox',
        packages = ['avalanchetoolbox'],
        version = '.2',
        description = 'Toolbox for identifying and describing branching processes, such as neuronal avalanches',
        author='Jeff Alstott',
        author_email = 'jeffalstott@gmail.com',
        url = 'https://github.com/jeffalstott/avalanchetoolbox',
        requires = ['powerlaw', 'scipy', 'numpy', 'sqlalchemy', 'h5py', 'matplotlib'],
        classifiers = [
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2.7',
            'Operating System :: OS Independent',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Physics',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Information Analysis',
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research'
            ]
        )
