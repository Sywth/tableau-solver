from src import sat, theory
from enum import Enum
import unittest


class SatOutput(Enum):
    UN_SATISFIABLE = 0
    SATISFIABLE = 1
    UNDECIDED = 2


def getOutput(line: str) -> SatOutput:
    tableau = theory(line)
    return SatOutput(sat([tableau]))


class TestZol(unittest.TestCase):
    def testcase1(self):
        out = getOutput("p")
        self.assertEqual(out, SatOutput.SATISFIABLE)

    def testcase2(self):
        out = getOutput("~p")
        self.assertEqual(out, SatOutput.SATISFIABLE)

    def testcase3(self):
        out = getOutput("(p/\\q)")
        self.assertEqual(out, SatOutput.SATISFIABLE)

    def testcase4(self):
        out = getOutput("(p=>p)")
        self.assertEqual(out, SatOutput.SATISFIABLE)

    def testcase5(self):
        out = getOutput("~(p=>q)")
        self.assertEqual(out, SatOutput.SATISFIABLE)

    def testcase6(self):
        out = getOutput("((p=>p)/\\(p=>q))")
        self.assertEqual(out, SatOutput.SATISFIABLE)

    def testcase7(self):
        out = getOutput("~(p=>p)")
        self.assertEqual(out, SatOutput.UN_SATISFIABLE)

    def testcase8(self):
        out = getOutput("~((p=>p)/\\(p=>p))")
        self.assertEqual(out, SatOutput.UN_SATISFIABLE)

    def testcase9(self):
        out = getOutput("(p/\\~p)")
        self.assertEqual(out, SatOutput.UN_SATISFIABLE)

    def testcase10(self):
        out = getOutput("~((p=>p)=>(q=>q))")
        self.assertEqual(out, SatOutput.UN_SATISFIABLE)

    def testcase11(self):
        out = getOutput("(p\\/r)")
        print(out)
        self.assertEqual(out, SatOutput.SATISFIABLE)

    def testcase12(self):
        out = getOutput("((p\\/q)/\\((p=>~p)/\\(~p=>p)))")
        self.assertEqual(out, SatOutput.UN_SATISFIABLE)

    def testcase13(self):
        out = getOutput("~((p\\/q)/\\((p=>~p)/\\(~p=>p)))")
        self.assertEqual(out, SatOutput.SATISFIABLE)


if __name__ == "__main__":
    unittest.main()
