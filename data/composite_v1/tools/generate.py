#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging
import argparse


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Generate a Synthetic Dataset', formatter_class=formatter)
    argparser.add_argument('entities', action='store', type=int, default=1024)

    args = argparser.parse_args(argv)

    nb_entities = args.entities

    for i in range(nb_entities):
        print('<a%d>\t<p>\t<b%d>' % (i, i))
        print('<b%d>\t<q>\t<c%d>' % (i, i))
        print('<a%d>\t<r>\t<c%d>' % (i, i))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
