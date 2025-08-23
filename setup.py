import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()
    

__version__ = "0.0.0"

REPO_NAME = "End_To_End_Machine_Learning_Project_MLOps"
AUTHOR_USER_NAME = "Juanda334"
SRC_REPO = "End-to-End-ML-Pipeline"
AUTHOR_EMAIL = "jd.valderrama@uniandes.edu.co"

setuptools.setup(
    name = SRC_REPO,
    version = __version__,
    author = AUTHOR_USER_NAME,
    author_email = AUTHOR_EMAIL,
    description = "A small python package for end to end ml project",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls = {
        "Bug Tracker": f"https:github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where = "src")
)