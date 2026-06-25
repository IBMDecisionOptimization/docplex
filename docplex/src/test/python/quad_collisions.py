from docplex.mp.quad import VarPair
from docplex.mp.model import Model
from private.timing import MyTimer


# compute the number of collisions for N variables with a combination function
def compute_vp_collision_rate(fn, nbvars):
    with Model() as m:
        xs = m.continuous_var_list(keys=nbvars)
        vps = [VarPair(x, y) for x in xs for y in xs if y.index < x.index]
        hsize = len(set(fn(vp, hash(vp.first), hash(vp.second)) for vp in vps))
        collisions = (len(vps) - hsize) / float(len(vps))
    return collisions


def hsum(vp, x, y):
    return x + y


def cantor(vp, x, y):
    return ((x + y) * (x + y + 1)) / 2 + y


def cantord(vp, x, y):
    # that's a cantor indexer, modified for diagonal
    # to return the squared variable hash
    if x == y:
        return x
    else:
        return cantor(x, y)


def bitwise(vp, f, s):
    h = (((f & 0xffffffff) ^ ((f >> 32) & 0xffffffff)) |
         ((s & 0xffffffff) ^ ((s >> 32) & 0xffffffff)) << 32)
    return h


def prod(vp, x, y):
    return x * y + y

def htuple(vp, x, y):
    return hash((vp.first, vp.second))


if __name__ == "__main__":

    sizes = [300, 700, 1000, 2000]
    # with MyTimer('hsum'):
    #     for sz in sizes:
    #         print('# varpair collisions for {0} vars is {1:.2f}'.format(sz, compute_vp_collision_rate(hsum, sz)))
    #
    # with MyTimer('cantord'):
    #     for sz in sizes:
    #         print('# varpair collisions for {0} vars is {1}'.format(sz, compute_vp_collision_rate(cantord, sz)))

    with MyTimer('htuple'):
        for sz in sizes:
            print('# varpair collisions for {0} vars is {1}'.format(sz, compute_vp_collision_rate(htuple, sz)))
