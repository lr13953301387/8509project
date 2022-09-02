import cdflib


def pre_process():
    output={}
    subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
    for subject in subjects:
        output[subject] = {}
