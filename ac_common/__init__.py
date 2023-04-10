#__all__ = ["is_notebook"]
from . import utils
from . import viz
from .classes import * # this means we can use Param instead of classes.Param #
#from . import classes
from .opt import * # this means we can use bayes_opt instead of opt.bayes_opt
#from . import opt
from . import acq_func

