eq = []
n = []
c = len(eq) - 1
size = 12
eqsize = 14


def filter_equations():
    import re
    for i in eq:
        eq[i] = re.sub(r'(-)?0(x|x²|x³)?', '', i)
        # eq[i].replace('0', '')
        # eq[i].replace('0x³', '')
        # eq[i].replace('0x²', '')
        # eq[i].replace('0x', '')
