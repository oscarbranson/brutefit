from setuptools import setup, find_packages
import brutefit

setup(name='brutefit',
      version=brutefit.__version__,
      description='Tools for finding an arbitrary multivariate polynomial that best fits some data.',
      url='https://github.com/oscarbranson/brutefit',
      author='Oscar Branson',
      author_email='oscarbranson@gmail.com',
      license='MIT',
      packages=find_packages(),
      keywords=['science', 'data', 'empiricism'],
      classifiers=['Development Status :: 4 - Beta',
                   'Intended Audience :: Science/Research',
                   'Programming Language :: Python :: 3 :: Only',
                   ],
      install_requires=['numpy',
                        'sklearn',
                        'pandas',
                        'sympy',
                        'tqdm'],
      zip_safe=True)
