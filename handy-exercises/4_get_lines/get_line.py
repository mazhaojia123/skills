def get_numbers_behind(a_string, filename):
    '''获取指定文件名为 filename 的某个单词后的数据, 返回一个list存储float类型的结果. 
    Args: 
        a_string (str): 某个特殊的单词 
        filename (str): the name of the text file. 

    Returns:
        list: A list of float numbers. If the col is not digits, the func return a None. 
    '''
    f = open(filename, "r")
    lines = f.readlines()
    result = []
    for aline in lines:
        line_lst = aline.split(' ')
        for i in range(len(line_lst)):
            if line_lst[i] == a_string and i+1 < len(line_lst):
                tmp = line_lst[i+1]
                tmp = tmp.replace('.', '')
                if tmp.isnumeric():
                    result.append(float(line_lst[i+1]))
                else:
                    result = None
                break # 只找到第一个数就行 

        if result == None:
            break
    f.close()
    return result

