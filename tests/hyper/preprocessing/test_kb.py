# -*- coding: utf-8 -*-

import unittest
from hyper.preprocessing import kb


class TestKnowledgeBaseParser(unittest.TestCase):
    def setUp(self):
        pass

    def test_kb_parser(self):
        lines = [
            's1 p1 s2',
            's2 p1 s2',
            's1 p2 s1'
        ]

        facts = list()
        for line in lines:
            subject, predicate, object = line.split()
            facts.append(kb.Fact(predicate_name=predicate, argument_names=[subject, object]))

        parser = kb.KnowledgeBaseParser(facts)

        self.assertTrue('s1' in parser.entity_index)
        self.assertTrue('s2' in parser.entity_index)
        self.assertTrue('p1' in parser.predicate_index)
        self.assertTrue('p2' in parser.predicate_index)
        self.assertTrue('p3' not in parser.predicate_index)


if __name__ == '__main__':
    unittest.main()
