from morebs2.pr2label import * 
import unittest

'''
py -m morebs2.tests.pr2label_test  
'''
class TestPR2LabelFunctions(unittest.TestCase):

    # normalized euclidean point distance, w/ reference?
    def test__probability_to_label__case_1(self):

        pr2label_vec = [(0.3,"FUCK"),(0.2,"YOUR"),(0.3,"PEOPLE."),(0.05,"IT'S"),(0.15,"TRUE.")] 

        pr_vec = [i * 0.05 for i in range(21)] 
        labels = [] 

        print("if you fuck me,tony... #Hispania") 
        for p in pr_vec: 
            p2 = probability_to_label(pr2label_vec,p)
            labels.append(p2) 

        assert labels == ['FUCK', 'FUCK', 'FUCK', 'FUCK', 'FUCK', 'FUCK', \
            'YOUR', 'YOUR', 'YOUR', 'YOUR', 'YOUR', \
            'PEOPLE.', 'PEOPLE.', 'PEOPLE.', 'PEOPLE.', 'PEOPLE.', 'PEOPLE.', \
            "IT'S", \
            'TRUE.', 'TRUE.', 'TRUE.']


if __name__ == "__main__":
    unittest.main()