#class Solution:
def matrixReshape(mat, r, c):
    """number of rows"""
    m = len(mat)
    """number of columns"""
    n = len(mat[0])
    """checking if conversion is possible and input matrix has data to traverse"""

    if m and n:
        if r * c != m * n:
            return mat
        flat_list = [item for row in mat for item in row]
        resh_list = []
        for i in range(0, r):
            if not len(resh_list):
                resh_list=[[flat_list.pop(0)]]
            elif len(resh_list) < i+1:
                resh_list.append([flat_list.pop(0)])
            while len(resh_list[i]) < c:
                resh_list[i].append(flat_list.pop(0))
        return resh_list

if __name__ == "__main__":
    # Graph using dictionaries
    mat=[[1, 2], [3, 4]]
    r=4
    c=1
    resh=matrixReshape(mat, r, c)
    print(resh)