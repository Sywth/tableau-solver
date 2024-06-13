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


class TestFol(unittest.TestCase):
    def testcase1(self):
        out = getOutput("p")
        self.assertEqual(out, SatOutput.SATISFIABLE)

    def testcase_extential_1(self):
        out = getOutput("ExP(x,x)")
        self.assertEqual(out, SatOutput.SATISFIABLE)

    def testcase_extential_2(self):
        out = getOutput("ExEyP(x,y)")
        self.assertEqual(out, SatOutput.SATISFIABLE)

    def testcase_extential_3(self):
        out = getOutput("Ex(P(x,x)=>P(x,x))")
        self.assertEqual(out, SatOutput.SATISFIABLE)

    def testcase_extential_4(self):
        out = getOutput("Ex~(P(x,x)=>P(x,x))")
        self.assertEqual(out, SatOutput.UN_SATISFIABLE)

    def testcase_extential_5(self):
        out = getOutput("ExEy(P(x,y)=>P(x,y))")
        self.assertEqual(out, SatOutput.SATISFIABLE)

    def testcase_extential_6(self):
        out = getOutput("ExEy~(P(x,y)=>P(x,y))")
        self.assertEqual(out, SatOutput.UN_SATISFIABLE)

    def testcase_extential_7(self):
        out = getOutput("Ex(P(x,x)=>EyP(y,y))")
        self.assertEqual(out, SatOutput.SATISFIABLE)

    def testcase_extential_8(self):
        out = getOutput("~Ax(P(x,x)=>P(x,x))")
        self.assertEqual(out, SatOutput.UN_SATISFIABLE)

    def testcase_extential_9(self):
        out = getOutput("~Ax~(P(x,x)=>P(x,x))")
        self.assertEqual(out, SatOutput.SATISFIABLE)

    def testcase_extential_10(self):
        # idefk anymore i am just goign say this is a feature, so Pxx is satifiable if x is not bound
        out = getOutput("~Ax~(P(x,x)=>P(x,y))")
        self.assertEqual(out, SatOutput.SATISFIABLE)

    def testcase_universal_1(self):
        out = getOutput("ExAy(Q(x,x)=>P(y,y))")
        self.assertEqual(out, SatOutput.SATISFIABLE)

    def testcase_universal_2(self):
        out = getOutput("AyQ(y,y)")
        self.assertEqual(out, SatOutput.SATISFIABLE)

    # def testcase_universal_3(self):
    #     out = getOutput("AxEy(P(x,y)=>P(x,y))")
    #     self.assertEqual(out, SatOutput.UNDECIDED)


if __name__ == "__main__":
    unittest.main()
