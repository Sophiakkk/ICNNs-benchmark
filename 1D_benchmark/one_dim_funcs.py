import numpy as np

# gramacy and lee function
def grammy_and_lee(x):
    return np.sin(10*np.pi*x) / (2*x) + (x - 1)**4


def dyhotomy(a, b, eps):
    root = None
    while abs(grammy_and_lee(b)-grammy_and_lee(a)) > eps:
        mid = (a+b) / 2
        if grammy_and_lee(mid) == 0 or abs(grammy_and_lee(mid)) < eps:
            root = mid
            break
        elif grammy_and_lee(a)*grammy_and_lee(mid) < 0:
            b = mid
        else:
            a = mid

    if root is None:
        print('Root not found')
    else:
        print(f'The root, according to the dichotomy method, is at the point x = {root}')
        return root