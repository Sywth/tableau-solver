from src.interpreter import *
from src import theory
import unittest


def getParsed(line: str) -> ParsedNode:
    tableau = theory(line)
    return tableau


class TestParser(unittest.TestCase):
    def test_literal_1(self):
        out = getParsed("p")
        self.assertEqual(out, ParsedPropositionLetter("p"))

    def test_literal_2(self):
        out = getParsed("q")
        self.assertEqual(out, ParsedPropositionLetter("q"))

    def test_conjunction_1(self):
        out = getParsed("(p/\\q)")
        self.assertEqual(
            out,
            ParsedConjunction(
                ParsedPropositionLetter("p"), ParsedPropositionLetter("q")
            ),
        )

    def test_disjunction_1(self):
        out = getParsed("(p\\/q)")
        self.assertEqual(
            out,
            ParsedDisjunction(
                ParsedPropositionLetter("p"), ParsedPropositionLetter("q")
            ),
        )

    def test_implication_1(self):
        out = getParsed("(p=>q)")
        self.assertEqual(
            out,
            ParsedImplication(
                ParsedPropositionLetter("p"), ParsedPropositionLetter("q")
            ),
        )

    def test_negated_literal_1(self):
        out = getParsed("~p")
        self.assertEqual(out, ParsedNegation(ParsedPropositionLetter("p")))

    def test_negated_conjunction_1(self):
        out = getParsed("~(p/\\q)")
        self.assertEqual(
            out,
            ParsedNegation(
                ParsedConjunction(
                    ParsedPropositionLetter("p"), ParsedPropositionLetter("q")
                )
            ),
        )

    def test_negated_disjunction_1(self):
        out = getParsed("~(p\\/q)")
        self.assertEqual(
            out,
            ParsedNegation(
                ParsedDisjunction(
                    ParsedPropositionLetter("p"), ParsedPropositionLetter("q")
                )
            ),
        )

    def test_predicate_2(self):
        self.assertRaises(Exception, getParsed, "P(x,y,z)")

    def test_fol_1(self):
        self.assertRaises(Exception, getParsed, "ExEy((Q(x,x)/\\Q(y,y))\\/)")


if __name__ == "__main__":
    unittest.main()
