try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Group Tractography Statistics',
    'author': 'David Qixiang Chen',
    'url': 'URL to get it at.',
    'download_url': 'Where to download it.',
    'author_email': 'qixiang.chen@gmail.com',
    'version': '0.1',
    'install_requires': ['vtk','nibabel','numpy'],
    'packages': ['gts'],
    'scripts': [],
    'name': 'gts'
}

setup(**config)