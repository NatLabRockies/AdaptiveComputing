#__all__ = ["is_notebook"]
from . import utils
from . import viz
from .classes import * # this means we can use Param instead of classes.Param #
#from . import classes
from .opt import * # this means we can use bayesOpt instead of opt.bayesOpt
#from . import opt
from . import acqFunc

