# classes.py
#########################################################
class Params:
    def __new__(cls): # make this class a singleton
        if not hasattr(cls, 'instance'):
            cls.instance = super(Params, cls).__new__(cls)
        return cls.instance

#########################################################
class Options:
    def __new__(cls): # make this class a singleton
        if not hasattr(cls, 'instance'):
            cls.instance = super(Options, cls).__new__(cls)
        return cls.instance
    # set the default options
    surrogateModel = 'gaussianProcess'
    acquFunc = 'EI'
    animation = False
    animation_dir = './movie'

### test code
if __name__ == "__main__":
    print('Checking that Params is a singleton: ')
    singleton = Params()
    new_singleton = Params()
    print(singleton is new_singleton)

    print('Checking that Options is a singleton: ')
    singleton = Options()
    new_singleton = Options()
    print(singleton is new_singleton)
