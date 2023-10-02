import numpy as np

# gramacy and lee function
def gramacy_and_lee(x):
    return np.sin(10*np.pi*x) / (2*x) + (x - 1)**4


def dyhotomy(a, b, eps):
    root = None
    while abs(gramacy_and_lee(b)-gramacy_and_lee(a)) > eps:
        mid = (a+b) / 2
        if gramacy_and_lee(mid) == 0 or abs(gramacy_and_lee(mid)) < eps:
            root = mid
            break
        elif gramacy_and_lee(a)*gramacy_and_lee(mid) < 0:
            b = mid
        else:
            a = mid

    if root is None:
        print('Root not found')
    else:
        print(f'The root, according to the dichotomy method, is at the point x = {root}')
        return root