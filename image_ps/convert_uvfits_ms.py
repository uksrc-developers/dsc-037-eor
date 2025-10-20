#!/usr/bin/env python3

import sys
import pyuvdata

def main():
    file = sys.argv[1]
    d = pyuvdata.UVData()
    d.read_uvfits(file)
    d.write_ms(file.replace('.uvfits', '.ms'))

if __name__ == '__main__':
    main()

