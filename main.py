from functionals.analysis import mbti_analysis


if __name__ == '__main__':
    # type 0 正常模式
    # type 4 消融4
    # type 6 消融6
    # type 7 消融7
    mbti_analysis(start=0, end=300, dataset='kaggle')
