from setuptools import setup

setup(name='tac-kbp-events',
      version='0.1',
      description=' Tools for TAC KBP Event Argument Extraction and Linking Shared Task',
      url='http://github.com/cgl/tac_kbp_events',
      author='cgl',
      author_email='cagilulusahin@gmail.com',
      license='MIT',
      packages=['events',
                #'sequencing',
      ],
      install_requires=[
          'nltk',
          'gensim',
          'sklearn',
      ],
      zip_safe=False)
