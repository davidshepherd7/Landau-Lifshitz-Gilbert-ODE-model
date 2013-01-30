from distutils.core import setup

setup(
    name='SimpleLLG',
    version='0.1.0',
    author='D.Shepherd'
    author_email='davidshepherd7@gmail.com'
    packages=['simplellg']
    scripts=[]
    url=''
    license='LICENSE.txt',
    description='A simple LLG solver    ... '
    long_description=open('README.txt').read(),
    install_requires=[
        "SciPy >= 0.10.1"
    ],
)
