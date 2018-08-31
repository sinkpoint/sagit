#!/usr/bin/env python


try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

#myreqs = parse_requirements('requirements.txt')

config = {
    'description': 'SAGIT: Selecive Automated Group Integrated Tractography',
    'author': 'David Qixiang Chen',
    'url': 'https://github.com/sinkpoint/gts',
    'download_url': 'https://github.com/sinkpoint/gts',
    'author_email': 'qixiang.chen@gmail.com',
    'version': '0.1',
    'install_requires': ['nibabel','numpy', 'scipy'],
    'packages': find_packages(),
    'name': 'sagit',
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    'entry_points': {
        'console_scripts': [
            'sagit_scalar_to_tracts=gts.scripts.gts_scalar_to_tracts:main',
            'sagit_tracts_to_density=gts.scripts.gts_tracts_to_density:main',
            'sagit_dwi_to_apm=gts.scripts.sh_to_apm:main',
            'sagit_fiber_to_table=gts.scripts.fiber_to_table:main',
            'sagit_fiber_stats=gts.scripts.fiber_stats:main',
            'sagit_fiber_stats_plot=gts.scripts.fiber_stats_plot:main',
            'sagit_fiber_stats_compare=gts.scripts.fiber_stats_compare:main',
            'sagit_fiber_stats_gp=gts.scripts.fiber_stats_gp:main',
            'sagit_fiber_stats_gp_classify=gts.scripts.fiber_stats_gp_classify:main',
            'sagit_nos_score=gts.meas.imagescore:main'
        ],
    },
}

setup(**config)
