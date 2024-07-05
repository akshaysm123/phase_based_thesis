class BatchIDX:
    def __init__(self, start, end, load, vstart, vend):
        self.start = start
        self.end = end
        self.loadCache = load
        self.saveCache = -1
        self.vStart = vstart
        self.vEnd = vend

def batch_organizer(batchsize, nframes, cachesize):
    batchsize -= 2*cachesize
    nbatches = nframes // batchsize + int(nframes % batchsize != 0)
    last_end = 0
    idxs = []
    for b in range(nbatches):
        start = b*batchsize
        end = min((b+1)*batchsize, nframes)

        vstart = start
        vend = end

        if b == 0:
            end += 2*cachesize
        elif b > 0 and b < (nbatches-1):
            start -= cachesize
            end += cachesize
        else:
            start -= batchsize - (end-start) 
            start -= 2*cachesize

        vstart -= start
        vend -= end
        if vend == 0:
            vend -= 1

        uncached =  last_end - start
        idxs.append(BatchIDX(start, end, uncached, vstart, vend))
        last_end = end

    for i in range(len(idxs) - 1):
        idxs[i].saveCache = -idxs[i+1].loadCache

    return idxs