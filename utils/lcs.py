def LCSubStr(X, Y):

    # Create a table to store lengths of
    # longest common suffixes of substrings.
    # Note that LCSuff[i][j] contains the
    # length of longest common suffix of
    # X[0...i-1] and Y[0...j-1]. The first
    # row and first column entries have no
    # logical meaning, they are used only
    # for simplicity of the program.

    # LCSuff is the table with zero
    # value initially in each cell
    m = len(X)
    n = len(Y)
    LCSuff = [[0 for k in range(n+1)] for l in range(m+1)]

    # To store the length of
    # longest common substring
    result = 0
    max_str = ""
    # Following steps to build
    # LCSuff[m+1][n+1] in bottom up fashion
    xidx = (0, 0)
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
            elif (X[i-1] == Y[j-1]):
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                if LCSuff[i][j] > result:
                    result = LCSuff[i][j]
                    max_str = X[i - result:i]
                    xidx = (i-result, i)
            else:
                LCSuff[i][j] = 0
    return result, max_str, xidx

def LCS(a, b):
    # generate matrix of length of longest common subsequence for substrings of both words
    lengths = [[0] * (len(b)+1) for _ in range(len(a)+1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])

    # read a substring from the matrix
    result = []
    j = len(b)
    xst = -1
    xen = 0
    for i in range(1, len(a)+1):
        if lengths[i][j] != lengths[i-1][j]:
            result.append(a[i-1])
            if xst < 0:
                xst = i-1
            xen = i

    return len(result), result, (xst, xen)
