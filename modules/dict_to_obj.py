
# usage: args = DictToObj(**args)
class DictToObj:
    def __init__(self, **entries):
        self.__dict__.update(entries)
