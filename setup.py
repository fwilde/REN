from distutils.core import setup

setup(name = 'REN',
      version = '0.1',
      description = 'ROI Extraction Net: supervised ROI extraction for microscope images using a CNN',
      author = 'Fabian Wilde',
      author_email = 'fabian.wilde@uni-greifswald.de',
      url = 'https://github.com/fwilde/REN,
      install = ['numpy', 'scipy', 'matplotlib', 'tensorflow', \
                 'read-roi', 'tqdm', 'skimage', 'seaborn']
     )
