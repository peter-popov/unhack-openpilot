import argparse


class ArgParserWrapper(object):
    """
    Wrapper class for argument parsing
    """
    def __init__(self, *args, **kwargs):
        self.parser = argparse.ArgumentParser(*args, **kwargs)
        self.required = self.parser.add_argument_group('required arguments')
        self.optional = self.parser.add_argument_group('optional arguments')

    def add_required_argument(self, *args, **kwargs):
        self.required.add_argument(*args, **kwargs)

    def add_optional_argument(self, *args, **kwargs):
        self.optional.add_argument(*args, **kwargs)

    def parse_args(self, args=None, namespace=None):
        return self.parser.parse_args(args, namespace)
