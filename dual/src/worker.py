import mouette as M
from .instance import Instance
from .common import Options,VerboseOptions

class Worker(M.Logger):
    """
    An abstract worker applying functions on a parametrization problem instance
    Base class for Initializer, Optimizer and ParamConstructor
    """

    def __init__(self, 
        name : str, 
        instance : Instance, 
        options : Options = Options(),
        verbose_options : VerboseOptions = VerboseOptions()):
        
        self.instance = instance
        self.options = options
        self.verbose_options = verbose_options
        M.Logger.__init__(self, name, self.verbose_options.logger_verbose)