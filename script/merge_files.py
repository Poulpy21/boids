#! /usr/bin/python

import os, re


def reglob(path, exp, invert=False):
    """glob.glob() style searching which uses regex
      :param exp: Regex expression for filename
      :param invert: Invert match to non matching files
    """

    m = re.compile(exp)
    if invert is False:
        res = [f for f in os.listdir(path) if m.search(f)]
    else:
        res = [f for f in os.listdir(path) if not m.search(f)]
    
    res = map(lambda x: "%s/%s" % ( path, x, ), res)
    return res


if __name__ == "__main__":
    outfilename = "data/boids_all.xyz" 
    filenames = reglob("data", "boids_[0-9]+.xyz")

    files = []
    i = 0

    outfile = open(outfilename, 'wb')
        
    for fname in filenames:
        print(fname)
    """    files[i] = open(fname)
        i += 1

    #for line in f:
    #    outfile.write(line)

    for f in files:
        f.close()
        """
