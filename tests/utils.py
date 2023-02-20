def pspec_context(s):
    def decorator(function):
        def _decorator():
            return function()

        _decorator.__doc__ = "\033[1m\033[93m" + s + "\033[0m"
        return _decorator

    return decorator
