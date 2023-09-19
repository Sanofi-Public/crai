import sys

from chimerax.core.commands import run


def get_vol_surf_thresh(thresh):
    """
    Return the volume and surface of the current mrc data at a given thresh
    """
    run(session, f"volume #1 level {thresh}")
    vol = run(session, "measure volume #1")
    surf = run(session, "measure area #1")
    return vol, surf


def get_ratio(thresh):
    vol, surf = get_vol_surf_thresh(thresh)
    res = float('inf') if vol == 0 else surf / vol
    return res


def brents(f, x0, x1, max_iter=50, tolerance=0.01):
    fx0 = f(x0)
    fx1 = f(x1)

    assert (fx0 * fx1) <= 0, "Root not bracketed"

    if abs(fx0) < abs(fx1):
        x0, x1 = x1, x0
        fx0, fx1 = fx1, fx0

    x2, fx2 = x0, fx0

    mflag = True
    steps_taken = 0

    while steps_taken < max_iter and abs(x1 - x0) > tolerance:
        fx0 = f(x0)
        fx1 = f(x1)
        fx2 = f(x2)

        if fx0 != fx2 and fx1 != fx2:
            L0 = (x0 * fx1 * fx2) / ((fx0 - fx1) * (fx0 - fx2))
            L1 = (x1 * fx0 * fx2) / ((fx1 - fx0) * (fx1 - fx2))
            L2 = (x2 * fx1 * fx0) / ((fx2 - fx0) * (fx2 - fx1))
            new = L0 + L1 + L2

        else:
            new = x1 - ((fx1 * (x1 - x0)) / (fx1 - fx0))

        if ((new < ((3 * x0 + x1) / 4) or new > x1) or
                (mflag == True and (abs(new - x1)) >= (abs(x1 - x2) / 2)) or
                (mflag == False and (abs(new - x1)) >= (abs(x2 - d) / 2)) or
                (mflag == True and (abs(x1 - x2)) < tolerance) or
                (mflag == False and (abs(x2 - d)) < tolerance)):
            new = (x0 + x1) / 2
            mflag = True

        else:
            mflag = False

        fnew = f(new)
        d, x2 = x2, x1

        if (fx0 * fnew) < 0:
            x1 = new
        else:
            x0 = new

        if abs(fx0) < abs(fx1):
            x0, x1 = x1, x0

        steps_taken += 1

    return x1, steps_taken


path = sys.argv[1]
target_value = float(sys.argv[2])
print(path)
run(session, f"open {path}")
# vol, sa = get_sa_thresh(1)
# print('RESULT BRENTS :', vol, sa)
# print('RESULT',0.933 - get_ratio(0))
# print('RESULT',0.933 - get_ratio(0.99))
x, step = brents(lambda t: target_value - get_ratio(t), 0, 1, max_iter=100, tolerance=0.01)
print('RESULT BRENTS :', x, step)
