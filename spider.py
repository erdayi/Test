import os
import csv
import requests
from time import sleep
from random import uniform
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings("ignore")

# ==============================
# 🛠️ 用户配置区（按需修改）
# ==============================
START_PAGE = 330           # ←←← 从第几页开始爬（例如上次断在45页，这里设46）
END_PAGE_LIMIT = None        # 可选：最大爬到多少页（如 100），None 表示不限制
CSV_FILE = '汽车投诉2025.csv'  # 输出文件名

# 请求头
HEADERS = {
    "accept": "application/json, text/javascript, */*; q=0.01",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "connection": "keep-alive",
    # 注意：cookie 可能会过期！建议定期更新或使用 Session 自动管理
    "cookie": "",
    "host": "www.aqsiqauto.com",  # ⚠️ 注意：requests 会自动设置 Host，一般不需要手动写
    "sec-ch-ua": '"Chromium";v="142", "Microsoft Edge";v="142", "Not_A Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0",
    "x-requested-with": "XMLHttpRequest"
}

# 字段清理函数
def clean_field(text):
    return text.strip().replace('\n', ' ').replace('\r', ' ') if text else ''

# 从 HTML 中提取总页数
def get_total_pages_from_html(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    last_link = soup.select_one("ul.yiiPager li.last a")
    if last_link and "page=" in last_link.get("href", ""):
        try:
            return int(last_link["href"].split("page=")[1].split("&")[0])
        except Exception:
            pass
    return None

# 获取下一页 URL
def get_next_page_url(soup):
    next_elem = soup.select_one("ul.yiiPager li.next a")
    if next_elem and next_elem.get("href"):
        href = next_elem["href"]
        if href.startswith("/"):
            return "https://www.aqsiqauto.com" + href
    return None

# 解析当前页的投诉记录
def parse_records(soup):
    tbody = soup.find("tbody", id="tb1")
    if not tbody:
        return []
    records = []
    for row in tbody.find_all("tr"):
        tds = row.find_all("td")
        if len(tds) < 7:
            continue

        comp_id = clean_field(tds[0].find("div").get_text() if tds[0].find("div") else tds[0].get_text())
        brand = clean_field(tds[1].get_text())
        series = clean_field(tds[2].get_text())
        model = clean_field(tds[3].get_text())
        summary = clean_field(tds[4].get_text())

        issue_td = tds[5]
        issue_divs = issue_td.find_all("div")
        if len(issue_divs) >= 2:
            main_issue = clean_field(issue_divs[0].get_text())
            sub_issue = clean_field(issue_divs[1].get_text())
        else:
            txt = clean_field(issue_td.get_text())
            if "—" in txt:
                parts = txt.split("—", 1)
                main_issue = parts[0].strip()
                sub_issue = parts[1].strip() if len(parts) > 1 else ""
            else:
                main_issue, sub_issue = txt, ""

        date = clean_field(tds[6].get_text())

        records.append({
            "投诉编号": comp_id,
            "投诉品牌": brand,
            "投诉车系": series,
            "投诉车型": model,
            "投诉简述": summary,
            "投诉问题": main_issue,
            "问题类型": sub_issue,
            "投诉日期": date
        })
    return records

# ==============================
# 🔁 主爬虫逻辑
# ==============================

# 打开 CSV（追加模式）
file_exists = os.path.isfile(CSV_FILE)
csv_file = open(CSV_FILE, mode='a', encoding='utf-8', newline='')
csv_writer = csv.DictWriter(csv_file, fieldnames=[
    '投诉编号', '投诉品牌', '投诉车系', '投诉车型', '投诉简述', '投诉问题', '问题类型', '投诉日期'
])
if not file_exists:
    csv_writer.writeheader()
    print("📝 首次运行，已创建 CSV 文件并写入表头。")

current_page = START_PAGE
total_pages = None
MAX_RETRIES = 3

try:
    while True:
        url = f"https://www.aqsiqauto.com/qichetousu.html?car_brand_id=0&car_series_id=0&page={current_page}&complaint_number=&complaint_status=3%2C4%2C5%2C7"
        print(f"正在爬取第 {current_page} 页: {url}")

        retries = 0
        success = False

        while retries < MAX_RETRIES:
            try:
                response = requests.get(url, headers=HEADERS, timeout=15)
                response.encoding = 'utf-8'

                # 检查是否被拦截（内容过短）
                if len(response.text) < 500:
                    raise Exception(f"响应内容过短 ({len(response.text)} 字节)，疑似反爬拦截")

                soup = BeautifulSoup(response.text, "html.parser")

                # 首次获取总页数
                if total_pages is None:
                    total_pages = get_total_pages_from_html(response.text)
                    if total_pages:
                        print(f"📌 检测到总页数: {total_pages}")
                    else:
                        total_pages = 715  # 默认兜底
                        print("⚠️ 无法获取总页数，使用默认值 715")

                # 解析数据 & 下一页链接
                records = parse_records(soup)
                next_url = get_next_page_url(soup)

                print(f"✅ 第 {current_page} 页保存 {len(records)} 条记录")

                for rec in records:
                    csv_writer.writerow(rec)

                # 决定是否继续
                if next_url:
                    current_page += 1
                else:
                    print("🔚 未找到下一页链接，爬取结束。")
                    success = True
                    break

                success = True
                break  # 成功则跳出重试循环

            except Exception as e:
                retries += 1
                wait_sec = uniform(5, 10)
                print(f"❌ 第 {current_page} 页出错 (尝试 {retries}/{MAX_RETRIES}): {e}")
                print(f"⏳ 等待 {wait_sec:.1f} 秒后重试...")
                sleep(wait_sec)

        if not success:
            print(f"💥 第 {current_page} 页多次失败，跳过并尝试下一页")
            current_page += 1

        # 安全兜底：防止无限循环
        max_allowed = END_PAGE_LIMIT if END_PAGE_LIMIT else (total_pages or 1000)
        if current_page > max_allowed:
            print(f"🛑 已达到设定上限（{max_allowed} 页），强制停止。")
            break

        # 正常请求间隔
        sleep(uniform(2, 4))

except KeyboardInterrupt:
    print("\n⚠️ 用户中断爬取。")
finally:
    csv_file.close()
    print("💾 CSV 文件已关闭。")
    print("🎉 爬取流程结束。")