# some extra parameter parsers

import argparse
import re

class DictParser(argparse.Action):

    def __init__(self, *args, **kwargs):

        super(DictParser, self).__init__(*args, **kwargs)
        self.local_dict = {}

    def __call__(self, parser, namespace, values, option_string = None):
        try:
            # for parsing list like 'a=0,b=[1,2,3],c=[4,5,6], d=255/4'
            for kv in re.findall(r'\w+=[-/\w.\*]+|\w+=\[.*?\]', values):
                if kv.find('[') >= 0 and values.find(']') >= 0:
                    k, v = kv.split('=')
                    v = v.replace('[', '').replace(']', '').split(',')
                    try:
                        self.local_dict[k] = [eval(_) for _ in v]
                    except:
                        self.local_dict[k] = [_ for _ in v]
                else:    
                    k, v = kv.split('=')
                    try:
                        self.local_dict[k] = eval(v)
                    except:
                        self.local_dict[k] = v
            print(self.local_dict)
            setattr(namespace, self.dest, self.local_dict)
        except:
            raise ValueError('Failed when parsing %s as dict' % values)

class ListParser(argparse.Action):

    def __init__(self, * args, **kwargs):

        super(ListParser, self).__init__(*args, **kwargs)
        self.local_list = []

    def __call__(self, parser, namespace, values, option_string = None):

        try:
            self.local_list = values.split(',')
            setattr(namespace, self.dest, self.local_list)
        except:
            raise ValueError('Failed when parsing %s as str list' % values)

class IntListParser(argparse.Action):

    def __init__(self, *args, **kwargs):

        super(IntListParser, self).__init__(*args, **kwargs)
        self.local_list = []

    def __call__(self, parser, namespace, values, option_string = None):

        try:
            self.local_list = list(map(int, values.split(',')))
            setattr(namespace, self.dest, self.local_list)
        except:
            raise ValueError('Failed when parsing %s as int list' % values)

class FloatListParser(argparse.Action):

    def __init__(self, *args, **kwargs):

        super(FloatListParser, self).__init__(*args, **kwargs)
        self.local_list = []

    def __call__(self, parser, namespace, values, option_string = None):

        try:
            self.local_list = list(map(float, values.split(',')))
            setattr(namespace, self.dest, self.local_list)
        except:
            raise ValueError('Failed when parsing %s as float list' % values)

class BooleanParser(argparse.Action):

    def __init__(self, *args, **kwargs):

        super(BooleanParser, self).__init__(*args, **kwargs)
        self.values = None

    def __call__(self, parser, namespace, values, option_string = None):

        try:
            self.values = False if int(values) == 0 else True
            setattr(namespace, self.dest, self.values)
        except:
            raise ValueError('Failed when parsing %s as boolean list' % values)