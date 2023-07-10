import argparse


def get_numbers_at_column(col, filename):
    '''获取指定文件名为 filename 的第 col 列数据, 返回一个list存储float类型的结果. 
    Args: 
        col (int): column index of the number, which is start from 0; 
        filename (str): the name of the text file. 

    Returns:
        list: A list of float numbers. If the col is not digits, the func return a None. 
    '''
    f = open(filename, "r")
    lines = f.readlines()
    result = []
    for aline in lines:
        line_lst = aline.split(' ')
        if len(line_lst) <= col: # 这行不够长，跳过
            continue
        tmp = line_lst[col]
        tmp = tmp.replace('.', '')
        if tmp.isnumeric():
            result.append(float(line_lst[col]))
        else:
            result = None
            break
    f.close()
    return result


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


if __name__ == "__main__":
    # process the cli argument
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="The name of the text file", type=str)
    parser.add_argument(
        "mode", help="1: get numbers from certain column; 2: get numbers behind certain string", type=int)
    parser.add_argument(
        "str", help="1: should be the column index(start from 0); 2: should be the certain string", type=str)
    parser.add_argument(
        "--csv", help="output as a csv file", action="store_false")

    args = parser.parse_args()

    if args.mode == 1:
        col = int(args.str)
        ret = get_numbers_at_column(col, args.filename)
    elif args.mode == 2:
        ret = get_numbers_behind(args.str, args.filename)

    print(ret)
    print(len(ret))
