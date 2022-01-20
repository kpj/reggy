# set version dunder variable
from importlib import metadata

from .main import RegReg
from .families import *
from .regularizers import *

__version__ = metadata.version("regreg")
