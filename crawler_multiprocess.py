# ####################
# GitHub爬虫（多线程）
# 
# 命令行：
# nohup python -u crawler_multiprocess.py [--server | --local]
# --server：在python 3.9以下版本(服务器)运行添加此选项
# --local：在python 3.9及以上版本(本地)运行添加此选项
# 
# 同一目录下文件：
# config.json：记录目前爬虫进度。爬取到第几个用户、用户id。
# git_crawler_log.json：爬虫日志。记录每次http请求的url、请求次数（若失败次数为0）、发起的进程号（pid）
# nohup.out：标准输出。记录爬虫进度、时间等。
# binary_files：存储爬下来的二进制文件
# pid：记录主进程号。结束爬虫命令：kill -s SIGINT [主进程id]
# git_token.txt：GitHub personal access token，需要使用此认证将api访问次数提升到5000次/小时。不用此认证限速60次/小时。
# 运行前需手动创建这些文件/目录
# ####################
import requests
import json
import argparse
import time
from retrying import retry
import logging
from multiprocessing import Process, Queue, Value
import os

parser = argparse.ArgumentParser()
parser.add_argument('--server', dest='run_on_server', action='store_true')
parser.add_argument('--local', dest='run_on_server', action='store_false')
parser.set_defaults(run_on_server=True)
args = parser.parse_args()

if args.run_on_server:#python3.9版本以下
    path = "/home/ubuntu/git_crawler/"
    root_logger= logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"{path}git_crawler_log2.json", 'a', 'utf-8') 
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s')) 
    root_logger.addHandler(handler)
else:#python3.9版本
    path = "C:/Users/hhjimhhj/Desktop/实习/git_crawler/"
    logging.basicConfig(filename=f"{path}git_crawler_log.json", encoding='utf-8', level=logging.INFO, filemode='a', format="%(asctime)s %(message)s")

#导入GitHub personal access token。写入请求头中。每个关于GitHub api的请求都需要。
with open(f"{path}git_token.txt", 'r') as f:
    token = f.read()
    headers = {'Authorization': "token " + token}

# ####################
# get_request函数封装了requests.get方法。使用get_request代替后者。
# 参数：
# url：url
# i：全局变量用于记录request.get的调用次数
# headers：请求头
# stop_max_attempt_number：最多尝试get的次数，默认为None，即一直尝试直到成功
# ####################
def get_request(url, i:Value, headers=None, stop_max_attempt_number=None):
    retry_count = 0
    @retry(stop_max_attempt_number=stop_max_attempt_number)
    def _get_request(url, i:Value, headers=None):
        nonlocal retry_count
        retry_count += 1
        response = requests.get(url, headers=headers, timeout=1)
        i.value += 1
        if i.value % 100 == 0:
            print(f"{i.value} requests sent")
        if "X-RateLimit-Remaining" in response.headers and int(response.headers["X-RateLimit-Remaining"]) == 0:
            reset_time = int(response.headers["X-RateLimit-Reset"])
            logging.info(f"rate limit reached, pid: {os.getpid()}")
            while time.time() <= reset_time:
                time.sleep(1)
            raise Exception
        logging.info(f"url: {url}, count: {retry_count}, pid: {os.getpid()}")
        return response
    return _get_request(url=url, headers=headers, i=i)
    
def crawler(q:Queue, crawl_from_organization_id:Value, org_count:Value, headers:dict, i:Value):
    while True:
        while q.qsize() == 0:
            logging.info(f"the queue is empty!, pid: {os.getpid()}")
            time.sleep(1)
        org = q.get()
        crawl_from_organization_id.value = org["id"]
        with open(f"{path}config.json", 'w') as config_file:
            json.dump({"crawl_from_organization_id": org["id"], "org_count": org_count.value}, config_file)
        try:
            repos = get_request(f"https://api.github.com/orgs/{org['login']}/repos", headers=headers, stop_max_attempt_number=5, i=i)
        except:
            logging.info(f"url: https://api.github.com/orgs/{org['login']}/repos, count: 0, pid: {os.getpid()}")
            continue
        for repo in repos.json():
            try:
                releases = get_request(f"https://api.github.com/repos/{org['login']}/{repo['name']}/releases", headers=headers, stop_max_attempt_number=5, i=i)
            except:
                logging.info(f"url: https://api.github.com/repos/{org['login']}/{repo['name']}/releases, count: 0, pid: {os.getpid()}")
                continue
            if len(releases.json()):
                if type(releases.json()) is list:
                    for asset in releases.json()[0]["assets"]:# 只下载最新的release版本
                        url = asset["browser_download_url"]
                        try:
                            binary_file = get_request(url, stop_max_attempt_number=5, i=i)
                        except:
                            logging.info(f"url: {url}, count: 0, pid: {os.getpid()},")
                            continue
                        with open(f"{path}binary_files/{url.split('/')[-1]}", 'wb') as f:
                            f.write(binary_file.content)
        org_count.value += 1
        if org_count.value % 10 == 0:
            print(f"{org_count.value} organizations checked, {time.localtime(time.time())}")

def get_orgs(q:Queue, crawl_from_organization_id:int, org_count:int, headers:dict, i:Value):
    print(f"crawling from id:{crawl_from_organization_id.value}({org_count.value}th)")
    r = None
    while r == None or "next" in r.links:
        if q.qsize() == 0:
            # 直接爬用户效率低，用户的仓库质量较低，且可能存在重复的fork
            r = get_request(f'https://api.github.com/organizations?per_page=100&since={crawl_from_organization_id.value}', headers=headers, i=i)
            for org in r.json():
                q.put(org)
        else:
            time.sleep(1)

if __name__ == "__main__":

    with open(f"{path}config.json", 'r') as f:
        config_args = json.load(f)
        crawl_from_organization_id = config_args["crawl_from_organization_id"]
        org_count = config_args["org_count"]
    with open(f"{path}pid", 'w') as f:# 存储主进程id
        f.write(str(os.getpid()))
    crawl_from_organization_id = Value('i', crawl_from_organization_id)
    org_count = Value('i', org_count)
    i = Value('i', 0)

    q = Queue()
    logging.info(f"存储二进制文件至：{path}binary_files/")

    pool = []
    pool.append(Process(target=get_orgs, args=(q, crawl_from_organization_id, org_count, headers, i)))
    pool.append(Process(target=crawler, args=(q, crawl_from_organization_id, org_count, headers, i)))
    pool.append(Process(target=crawler, args=(q, crawl_from_organization_id, org_count, headers, i)))
    for p in pool:
        p.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:# 用kill -s SIGINT [主进程id]结束程序
        for p in pool:
            p.terminate()
    for p in pool:
        p.join()