import numpy as np

h_message = '''
>>> You can use "min" or "max" to control the minimum and maximum values.
>>> constant
y = c
>>> linear
y = start_v + x * slope
>>> exp / exponential
y = start_v * power ** (x / interval)
>>> jump
y = start_v * power ** (max(idx - min_jump_pt + jump_freq, 0) // jump_freq)
'''


def continuous_seq(*args, **kwargs):
    '''
    >>> return a float to float mapping
    '''
    name = kwargs['name']
    max_v = kwargs['max'] if 'max' in kwargs else np.inf
    min_v = kwargs['min'] if 'min' in kwargs else -np.inf

    if name.lower() in ['h', 'help']:
        print(h_message)
        exit(0)
    elif name.lower() in ['constant', 'const']:
        start_v = float(kwargs['start_v'])
        base_func = lambda x: np.clip(start_v, a_min = min_v, a_max = max_v)
    elif name.lower() in ['linear',]:
        start_v = float(kwargs['start_v'])
        slope = float(kwargs['slope'])
        base_func = lambda x: np.clip(start_v + x * slope, a_min = min_v, a_max = max_v)
    elif name.lower() in ['exp', 'exponential',]:
        start_v = float(kwargs['start_v'])
        power = float(kwargs['power'])
        interval = int(kwargs['interval']) if 'interval' in kwargs else 1
        base_func = lambda x: np.clip(start_v * power ** (x / float(interval)), a_min = min_v, a_max = max_v)
    elif name.lower() in ['jump',]:
        start_v = float(kwargs['start_v'])
        power = float(kwargs['power'])
        min_jump_pt = int(kwargs['min_jump_pt'])
        jump_freq = int(kwargs['jump_freq'])
        base_func = lambda x: np.clip(start_v * power ** (max(x - min_jump_pt + jump_freq, 0) // jump_freq), a_min = min_v, a_max = max_v)
    elif name.lower() in ['cos', 'cosine']:
        start_v = float(kwargs['start_v'])
        alpha = float(kwargs['alpha']) if 'alpha' in kwargs else 0.0
        circle = float(kwargs['circle'])
        base_func = lambda x: np.clip(start_v * (alpha + (1 - alpha) / 2 * (1 + np.cos(np.pi * x / circle))), a_min = min_v, a_max = max_v)
    elif name.lower() in ['cyclic']:
        epochs_period = kwargs.get('epochs_period', [0, 100*2//5, 100*4//5, 100])
        lr_period = kwargs.get('lr_period', [0, 0.1, 0.005, 0])
        # base_func = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs], [0, 0.1, 0.005, 0])[0]
        base_func = lambda t: np.interp(t, epochs_period, lr_period)
    else:
        raise ValueError('Unrecognized name: %s'%name)

    if 'segs' in kwargs:
        segs = kwargs['segs'].split(':')
        segs = list(sorted(map(float, segs)))
        if 'seg_ratio' in kwargs:
            ratios = kwargs['seg_ratio'].split(':')
            ratios = list(map(float, ratios))
            assert len(ratios) >= len(segs), 'The length of segs and seg_ratio should be the same'
        else:
            ratios = [1.,] * len(segs)

        base_len = segs[1] - segs[0]
        def func(x):
            for seg_idx, seg_pt in enumerate(segs):
                if seg_pt > x:
                    if seg_idx == 0:
                        return base_func(segs[0])
                    last_pt = segs[seg_idx - 1]
                    prop = (x - last_pt) / (seg_pt - last_pt)
                    return base_func(segs[0] + base_len * prop) * ratios[seg_idx - 1]
            return base_func(segs[1] - 1e-6) * ratios[len(segs) - 1]
        return func
    else:
        return base_func

def discrete_seq(*args, **kwargs):
    '''
    >>> return a list of values
    '''
    name = kwargs['name']
    func = continuous_seq(*args, **kwargs)

    pt_num = int(kwargs['pt_num'])
    return [func(idx) for idx in range(pt_num)]

