from functionals.analysis import mbti_analysis


if __name__ == '__main__':
    # type 0 正常模式
    # type 4 消融4
    # type 6 消融6
    # type 7 消融7
    # type 8 消融8 (双专家系统)
    # type 9 消融9 (COT思维模式)
    mbti_analysis(start=0, end=50, dataset='pand', types=9)
    # mbti_analysis(start=50, end=100, dataset='kaggle', types=8)
