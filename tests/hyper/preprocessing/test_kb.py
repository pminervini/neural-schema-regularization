# -*- coding: utf-8 -*-

import unittest
from hyper.preprocessing import kb

import logging


class TestKnowledgeBaseParser(unittest.TestCase):
    def setUp(self):
        pass

    def test_kb_parser(self):
        facts = [
            's1 p1 s2',
            's2 p1 s2',
            's1 p2 s1'
        ]
        parser = kb.KnowledgeBaseParser([fact.split() for fact in facts])

        self.assertTrue('s1' in parser.symbol_index)
        self.assertTrue('s2' in parser.symbol_index)
        self.assertTrue('p1' in parser.symbol_index)
        self.assertTrue('p2' in parser.symbol_index)
        self.assertTrue('p3' not in parser.symbol_index)


if __name__ == '__main__':
    unittest.main()