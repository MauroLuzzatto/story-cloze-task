"""Setup module for project."""

from setuptools import setup, find_packages

setup(
        name='NLU project2: story clozed task',
        version='0.1',
        description='Code for Story Clozed Task in the Natural Language Understanding course.',

        author= ['Mauro Luzzatto', 'Dario Kneubuehler', 'Thomas Brunschwiler'],
        author_email= ['maurol@student.ethz.ch', 'kneubuehler@lem.ee.ethz.ch','ZRLTBR@ch.ibm.com'],

        packages=find_packages(exclude=[]),
        python_requires='>=3.5',
        install_requires=[
			# Add external libraries here.
            'numpy',
            'tensorflow',
            'json',
            'sklearn',
            'nltk',
            'tensorflow_hub',
            're'
        ],
)
