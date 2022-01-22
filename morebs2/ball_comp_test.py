from .ball_comp import *
from .message_streamer import *

def sample_BallComp_1():
    return -1

def test__BallComp__sample_1_sample_data_1():

    maxBalls = 20
    maxRadius = 5.0
    td = test_data_1()

    # TODO: delete k
    bc = BallComp(maxBalls,maxRadius,5,True)

    #bc.conduct_decision(td[0])

    for t in td:
        bc.conduct_decision(t)
    return

def test__BallComp__sample_1_sample_data_2():

    maxBalls = 20
    maxRadius = 5.0
    td = test_data_2()

    # TODO: delete k
    bc = BallComp(maxBalls,maxRadius,5,2)

    for t in td:
        bc.conduct_decision(t)

    print("********************")

    print("BALLS ", len(bc.balls))
    for k,v in bc.balls.items():
        print("k ",k)
        print(v)
        print()
    return

def test__BallComp__sample_1_sample_data_3():

    maxBalls = 20
    maxRadius = 5.0
    td = test_data_3()

    # TODO: delete k
    bc = BallComp(maxBalls,maxRadius,5,True)
    for t in td:
        bc.conduct_decision(t)

    print("********************")
    print("BALLS ", len(bc.balls))
    for k,v in bc.balls.items():
        print("k ",k)
        print(v)
        print()
    return

def test__BallComp__sample_1_sample_data_4():

    maxBalls = 5
    maxRadius = 20.0

    vh = ViolationHandler1(15,80.0)

    filePath = "indep/ballcomp_sample_data_4.txt"
    ms = MessageStreamer(filePath,readMode = 'r')
    bc = BallComp(maxBalls,maxRadius,5,vh,2)

    q = 20
    s = 0
    while ms.stream() and q > 0:
        for t in ms.blockData:
            if bc.conduct_decision(t) != -1:
                s += 1
        q -= 1

    print("********************")
    print("BALLS ", len(bc.balls))
    s_ = 0
    for k,v in bc.balls.items():
        print(v)
        print()
        s_ += v.data.newData.shape[0]
    print("*********************")
    print("number of ball points {} actual {}".format(s_,s))
    return

if __name__ == "__main__":
    test__BallComp__sample_1_sample_data_4()
