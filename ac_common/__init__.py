#__all__ = ["is_notebook"]
#from . import utils
#from . import viz
# .classes (instead of . classes) allows us to use Param instead of classes.Param #
from .classes import Param, ModelOptions, VizOptions, BoOptions 
from .model import Model
from . import static_sampling
#from . import bo
#from . import acq_func

