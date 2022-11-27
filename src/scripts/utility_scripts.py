def create_closure(data):
    def closure():
        return data
    return closure

def create_closure_func(func, *args):
    def closure():
        return func(*args)
    return closure