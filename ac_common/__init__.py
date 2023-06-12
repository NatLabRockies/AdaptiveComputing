#__all__ = ["is_notebook"]
from . import utils
from . import viz
from .classes import * # this means we can use Param instead of classes.Param #
from .model import * # this means we can use model instead of model.Model
from . import static_sampling
from . import bo
from . import acq_func

