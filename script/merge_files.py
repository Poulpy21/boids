#! /usr/bin/python

import os, re, sys


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
    if not len(filenames):
        print("No input files found !")
        sys.exit(0)

    print("Using " + ", ".join(filenames) + " to produce " + outfilename)

    files = []
    blocks = []
    nBoids = []

    outfile = open(outfilename, 'wb')
    for fname in filenames:
        f = open(fname)
        files.append(f)
        l = [block.split("\n") for block in f.read().split("\n\n")]
        l[0] = l[0][1:] # Remove first empty line of the file
        blocks.append(l)
        nBoids.append([int(b[0]) for b in l])

    for i in range(0,len(blocks[0])):
        n = 0
        for file_n in nBoids:
            n += file_n[i]
        outfile.write("\n"+ str(n) +"\n")
        for file_blocks in blocks:
            block = file_blocks[i]
            if len(block) >= 2:
                outfile.write("\n".join(block[1:]) + "\n")

    outfile.close()
    for f in files:
        f.close()
