from .glac import GLACAgent

def make_agent(algo: str, **kwargs):
    if algo == 'GLAC':
        return GLACAgent(**kwargs)
    else:
        raise ValueError(f'Unknown algorithm: {algo}')
    


def get_train(algo: str):
    if 'GLAC' in algo:
        from .glac import train as train
    
    else:
        raise ValueError(f'Unknown algorithm: {algo}')
    return train