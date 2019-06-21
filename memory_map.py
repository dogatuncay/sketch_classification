
import os
import mmap

class MemoryMap:
    def __init__(self, path, filename):
        filepath = os.path.join(path, filename + '.csv')
        self.size = os.stat(filepath).st_size
        self.fileno = os.open(filepath, os.O_RDONLY)
        self.memmap = mmap.mmap(self.fileno, self.size, access=mmap.ACCESS_READ)
        
    def __iter__(self):
        self.pos = 0
        return self

    def __next__(self):
        if self.pos < self.size:
            byte = self.memmap[self.pos]
            self.pos += 1
            return byte
        else:
            raise StopIteration
    
    def close(self):
        self.memmap.close()
        os.close(self.fileno)
