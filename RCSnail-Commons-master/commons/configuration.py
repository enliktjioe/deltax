
class Configuration:
    def __init__(self, argument_map):
        self.map = argument_map

    def exists(self, name):
        if name in self.map:
            return True
        else:
            return False

    def __getattr__(self, name):
        if name in self.map:
            return self.map[name]
        elif "enabled" in name:
            # enabled was not in the map, 
            # return default value False instead of raising exception
            return False
        else:
            raise AttributeError(name)
