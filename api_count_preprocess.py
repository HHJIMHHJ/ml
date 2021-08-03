# ####################
# 这段代码仅在本地环境（python 3.9.6）上运行过，未在服务器上运行过
# 这段代码用于预处理api_count_train.py所需的数据
# 拿到新数据的处理顺序应该是delete->count->stat
# 这段代码命令行形式如下：
# -f count|delete|stat [-m apistats|raw] [-i read_file/folder] [-o out_file]
# -f：选择功能
# -m：使用json文件中apistats记录的api数量还是重新按api调用序列计数
# -i：输入文件夹/文件夹，注意这个option可以接受多个参数，因此返回的是一个列表
# -o：输出文件
# 具体功能介绍在实现该功能的代码部分
# ####################

import os
import json
import argparse
import pandas as pd

path = "C:/Users/hhjimhhj/Desktop/实习/20000+/report"#默认路径（修改成自己的路径或者在命令行中指定路径）
parser = argparse.ArgumentParser()
parser.add_argument('-f')
parser.add_argument('-m')
parser.add_argument('-i', nargs='*')
parser.add_argument('-o')
args = parser.parse_args()

# ####################
#这个函数用于去除api名称中的"w", "A", "Ex", "ExW", "ExA"后缀
# ####################
def strip_suffixes(s:str)->str:
    if s.endswith("W") or s.endswith("A"):
        s = s[:-1]
    if s.endswith("Ex"):
        s = s[:-2]
    return s

# ####################
# count功能实现：用于统计所有文件中的api调用次数
# 命令行形式：
# -f count -m raw|apistats -i folder -o output_filename
# -m：使用json文件中apistats记录的api数量还是重新按api调用序列计数，两者有一点细微的差别
# -i：输入文件夹/文件夹，注意这个option可以接受多个参数，因此返回的是一个列表
# -o：输出文件
# ####################
if args.f == "count":
    if args.i:
        path = args.i[0]
    files = os.listdir(path)
    i = 0
    api_calls = {}
    for file in files:
        i = i + 1
        if (i % 1000) == 0:
            print(i)
        f = open(path + "/" + file, 'rb')
        s = json.load(f)
        if "data" in s:
            s = s["data"]
        # 从原始的api调用序列中统计api调用次数
        if args.m == "raw":
            for process in s["behavior"]["processes"]:
                for call in process["calls"]:
                    api = call["api"]
                    api = strip_suffixes(api)
                    if api in api_calls:
                        api_calls[api] += 1
                    else:
                        api_calls[api] = 1
        # 从原文件中的apistat中获取api调用次数
        else:
            for key in s["behavior"]["apistats"]:
                for key1 in s["behavior"]["apistats"][key]:
                    call_count = s["behavior"]["apistats"][key][key1]
                    key1 = strip_suffixes(key1)
                    if key1 in api_calls:
                        api_calls[key1] += call_count
                    else:
                        api_calls[key1] = call_count
    # 排序方便观察规律
    api_calls = dict(sorted(api_calls.items(), key = lambda item:item[1]))
    if args.o:
        save_file = open(args.o, 'w')
    else:
        save_file = open(path + "/../api_call_count.json", 'w')
    json.dump(api_calls, save_file, indent=1)
    save_file.close()

# ####################
# stat功能实现：分别统计出每个程序分别调用api的数量，用于训练神经网络
# 命令行形式：
# -f stat -i list_of_paths -o output_path
# -i：输入文件夹/文件夹，注意这个option可以接受多个参数，因此返回的是一个列表
# -o：输出文件：separate_count.csv是对每个程序的api调用数量的统计，是2维的pandas.DataFrame形式，一行为一个程序调用不同api的统计，
# 另外还有一个label属性，用来表示该样本是黑/白样本，这个文件就是预处理完成的数据文件；separate_count_stat.csv是对separate_count.csv调用pandas中describe函数的结果
# ####################
elif args.f == "stat":
    # 默认输入、输出路径，可在命令行中传入不同路径
    in_path_list = ["C:/Users/hhjimhhj/Desktop/实习/80+/report", "C:/Users/hhjimhhj/Desktop/实习/20000+/report"]
    out_path = "C:/Users/hhjimhhj/Desktop/实习/python_scripts"
    if args.i:
        in_path_list = args.i
    if args.o:
        out_path = args.o
    api_calls = {}
    files = []
    #这里两个raw_count.json文件就是count功能中输出的文件
    api_dict = json.load(open("C:/Users/hhjimhhj/Desktop/实习/20000+/raw_count.json", 'rb')) | json.load(open("C:/Users/hhjimhhj/Desktop/实习/80+/raw_count.json", 'rb'))
    api_dict['label'] = 0
    api_list = [key for key in api_dict]
    api_call_map = pd.DataFrame(columns=api_list)
    for path in in_path_list:
        files.extend([(path + '/' + dir, in_path_list.index(path)) for dir in os.listdir(path)])

    i = 0
    for file in files:
        for key in api_dict:
            api_dict[key] = 0
        if (i % 1000) == 0:
            print(i)
        f = open(file[0], 'rb')
        s = json.load(f)
        if "data" in s:
            s = s["data"]
        for process in s["behavior"]["processes"]:
            for call in process["calls"]:
                api = call["api"]
                api = strip_suffixes(api)
                api_dict[api] += 1
        #label==1是黑样本；label==0是白样本
        api_dict['label'] = file[1]
        api_call_map.loc[i] = api_dict
        i = i + 1
    api_call_map.to_csv(out_path + "/separate_count.csv", index=False)
    api_call_map.describe().to_csv(out_path + "/separate_count_stat.csv")

# ####################
# delete功能实现：删除具有空的api调用序列的文件
# 命令行形式：
# -f delete -i path_of_files_to_delete
# -i：输入文件夹，注意这个option可以接受多个参数，因此返回的是一个列表
# ####################
elif args.f == "delete":
    if args.i:
        path = args.i[0]
    files = os.listdir(path)
    i = 0
    api_calls = {}
    for file in files:
        i = i + 1
        if (i % 1000) == 0:
            print(i)
        f = open(path + "/" + file, 'rb')
        s = json.load(f)
        if "data" in s:
            s = s["data"]
        if len(s["behavior"]["processes"]) == 0:
            f.close()
            os.remove(path + "/" + file)
        else:
            for process in s["behavior"]["processes"]:
                if process["calls"] == None:
                    f.close()
                    os.remove(path + "/" + file)
                    break