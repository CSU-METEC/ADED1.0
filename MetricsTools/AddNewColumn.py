def addNewColumn(df, NewcolumnHeader, scaleFactor, l, m, n, *args):
    #Todo: This function is ammendable based on the what you want to achieve
    df.insert(len(df.columns), NewcolumnHeader, None)
    argslist = []
    for arg in args:
        argslist.append(arg)
    for ID, eRow in df.iterrows():
        if float(eRow[argslist[0]]) == float(0) and float(eRow[argslist[1]]) != float(0):
            df.loc[ID, NewcolumnHeader] = (eRow[argslist[0]]) * (scaleFactor**m) * (float(eRow[argslist[1]])**n)
        elif float(eRow[argslist[0]]) != float(0) and float(eRow[argslist[1]]) == float(0):
            df.loc[ID, NewcolumnHeader] = (eRow[argslist[0]]**l) * (scaleFactor**m) * float(eRow[argslist[1]])
        else:
            df.loc[ID, NewcolumnHeader] = (eRow[argslist[0]]**l)*(scaleFactor**m)*(float(eRow[argslist[1]])**n)
    return df