try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

#myreqs = parse_requirements('requirements.txt')

config = {
    'description': 'SAGIT: Selecive Automated Group Integrated Tractography',
    'author': 'David Qixiang Chen',
    'url': 'https://github.com/sinkpoint/gts',
    'download_url': 'https://github.com/sinkpoint/gts',
    'author_email': 'qixiang.chen@gmail.com',
    'version': '0.1',
    'install_requires': ['nibabel','numpy', 'scipy'],
    'packages': ['gts'],
    'name': 'gts',
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    'entry_points': {
        'console_scripts': [
            'gts_scalar_to_tracts=gts.scripts.gts_scalar_to_tracts:main',
            'gts_tracts_to_density=gts.scripts.gts_tracts_to_density:main',
        ],
    },    
}

setup(**config)