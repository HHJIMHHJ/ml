import requests
import json
import argparse
import time
from retrying import retry
import logging
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=int, help="step")
parser.add_argument('-s', nargs='*', help="run on server")
args = parser.parse_args()
if args.s != None:
    path = "/home/ubuntu/git_crawler/"
    root_logger= logging.getLogger()
    root_logger.setLevel(logging.INFO) # or whatever
    handler = logging.FileHandler(f"{path}git_crawler_log.json", 'a', 'utf-8') # or whatever
    handler.setFormatter(logging.Formatter('%(message)s')) # or whatever
    root_logger.addHandler(handler)
else:
    path = "C:/Users/hhjimhhj/Desktop/实习/git_crawler/"
    logging.basicConfig(filename=f"{path}git_crawler_log.json", encoding='utf-8', level=logging.INFO, filemode='a', format="%(message)s")

with open(f"{path}config.json", 'r') as f:
    config_args = json.load(f)
    crawl_from_organization_id = config_args["crawl_from_organization_id"]
    org_count = config_args["org_count"]

print(f"存储二进制文件至：{path}binary_files/")

token = open(f"{path}git_token.txt", 'r').read()
headers = {'Authorization': "token " + token}
i = 0
def get_request(url, headers=None, stop_max_attempt_number=None):
    global retry_count
    retry_count = 0
    @retry(stop_max_attempt_number=stop_max_attempt_number)
    def _get_request(url, headers=None):
        global retry_count
        retry_count += 1
        response = requests.get(url, headers=headers, timeout=1)
        global i
        i += 1
        if i % 100 == 0:
            print(f"{i} requests sent")
        if "X-RateLimit-Remaining" in response.headers and int(response.headers["X-RateLimit-Remaining"]) == 0:
            reset_time = int(response.headers["X-RateLimit-Reset"])
            while time.time() <= reset_time:
                time.sleep(10)
        logging.info(f"{{\"url\": {url}, \"count\": {retry_count}}},")
        return response
    return _get_request(url=url, headers=headers)
    
if args.f == 0:
    r = None
    while r == None or "next" in r.links:
    
        r = get_request(f'https://api.github.com/organizations?per_page=100')
        # with open(f"{path}organizations.json", mode='w', encoding="utf-8") as organizations:
        #     json.dump(r.json(), organizations)

        for org in r.json():
            try:
                repos = get_request(f"https://api.github.com/orgs/{org['login']}/repos", stop_max_attempt_number=5)
            except:
                print(f"{i} requests sent")
                print(repos.json())
                print(repos.headers)
                logging.info(f"{{\"url\": https://api.github.com/orgs/{org['login']}/repos, \"count\": 0}},")
                continue

elif args.f == 1:
    r = get_request(f'https://api.github.com/organizations?per_page=100', headers=headers)
    print(r.headers)
    print(type(r.headers["X-RateLimit-Remaining"]))
    print(type(r.headers["X-RateLimit-Reset"]))
    print(int(r.headers["X-RateLimit-Reset"]))
    print(time.time() <= int(r.headers["X-RateLimit-Reset"]))

elif args.f == 2:
    retry_count = 0
    # org_list = {}
    # with open(f"{path}organizations.json", mode='w', encoding="utf-8") as organizations:
    #     raw_str = organizations.read()
    #     if len(raw_str):
    #         org_list = json.loads(raw_str)

    # r = get_request(f'https://api.github.com/organizations?per_page=100&since={crawl_from_organization_id}', headers=headers)
    # with open(f"{path}organizations.json", mode='w', encoding="utf-8") as organizations:
    #     json.dump(r.json(), organizations)
    print(f"crawling from id:{crawl_from_organization_id}({org_count}th)")
    r = None

    while r == None or "next" in r.links:
    
        # 直接爬用户效率低，用户会fork很多repo，导致重复的内容
        r = get_request(f'https://api.github.com/organizations?per_page=100&since={crawl_from_organization_id}', headers=headers)
        # with open(f"{path}organizations.json", mode='w', encoding="utf-8") as organizations:
        #     json.dump(r.json(), organizations)

        for org in r.json():
            try:
                repos = get_request(f"https://api.github.com/orgs/{org['login']}/repos", headers=headers, stop_max_attempt_number=5)
            except:
                logging.info(f"{{\"url\": https://api.github.com/orgs/{org['login']}/repos, \"count\": 0}},")
                continue
            for repo in repos.json():
                try:
                    releases = get_request(f"https://api.github.com/repos/{org['login']}/{repo['name']}/releases", headers=headers, stop_max_attempt_number=5)
                except:
                    logging.info(f"{{\"url\": https://api.github.com/repos/{org['login']}/{repo['name']}/releases, \"count\": 0}},")
                    continue
                if len(releases.json()):
                    if type(releases.json()) is list:
                        for asset in releases.json()[0]["assets"]:# 只下载最新的release版本
                            url = asset["browser_download_url"]
                            try:
                                binary_file = get_request(url, stop_max_attempt_number=5)
                            except:
                                logging.info(f"{{\"url\": {url}, \"count\": 0}},")
                                continue
                            with open(f"{path}binary_files/{url.split('/')[-1]}", 'wb') as f:
                                f.write(binary_file.content)
            org_count += 1
            if org_count % 10 == 0:
                print(f"{org_count} organizations checked")
            with open(f"{path}config.json", 'w') as config_file:
                json.dump({"crawl_from_organization_id": org["id"], "org_count": org_count}, config_file)
            crawl_from_organization_id = org["id"]
            
    # json.dump(r.json(), open("C:/Users/hhjimhhj/Desktop/实习/git_crawler/repos.json", mode='w', encoding="utf-8"))
elif args.f == 3:
    t1 = time.perf_counter()
    r = requests.get('https://api.github.com/repos/pressly/sup/releases/assets/6959525', headers={'Accept': "application/octet-stream"})
    open("./test", 'wb').write(r.content)
    t2 = time.perf_counter()
    r = requests.get('https://github.com/pressly/sup/releases/download/v0.5.3/sup-darwin64')
    open("./test", 'wb').write(r.content)
    t3 = time.perf_counter()
    print(f"{t3 - t2} {t2 - t1}")

elif args.f == 4:
    r = requests.get('https://api.github.com/orgs/TheOpenSpaceSociety/repos')
    print(r.headers)
elif args.f == 5:
    print(time.localtime(1627886641))
    print(time.time())
