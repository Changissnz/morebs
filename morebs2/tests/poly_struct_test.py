from morebs2 import poly_struct
import unittest
import numpy as np

class TestPolyClasses(unittest.TestCase):

    def test__SPoly__apply(self):

        # poly case 1
        sp = poly_struct.SPoly(np.array([12.0,0.0,3.0,1.0,2.0]))

        #   x1
        v1 = sp.apply(3.0)
        print(v1)
        assert v1 == 999 + 3 + 2, "incorrect SPoly case 1.1"

        v2 = sp.apply(0.0)
        assert v2 == 2.0, "incorrect SPoly case 1.2"

    def test__ISPoly__apply(self):

        s = poly_struct.ISPoly(3.0)
        v1 = s.apply(np.array([12.0,0.0,3.0,1.0,2.0]))

        assert v1 == 999 + 3 + 2, "incorrect case 1.1"


if __name__ == "__main__":
    unittest.main()
    print()
