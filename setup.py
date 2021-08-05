import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='self_supervised',
    version='0.0.1',
    author='Ravid Shwartz-Ziv',
    #author_email='mike_huls@hotmail.com',
    description='Self supervised learning ',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ravidziv/self-supervised-learning',
    project_urls = {
        "Bug Tracker": "https://github.com/ravidziv/self-supervised-learning/issues"
    },
    license='MIT',
    packages=['self_supervised'],
    install_requires=['tensorflow'],
)