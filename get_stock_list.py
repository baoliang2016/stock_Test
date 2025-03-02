import baostock as bs
import pandas as pd
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='stock_codes.log'
)

def get_all_a_stocks(date=None):
    """
    获取指定日期的所有A股股票代码
    如果 date 为 None，则使用当前日期
    """
    # 如果未指定日期，使用当前日期
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    # 登录 Baostock
    lg = bs.login()
    if lg.error_code != '0':
        logging.error(f"Baostock login failed: {lg.error_msg}")
        raise Exception(f"Baostock login failed: {lg.error_msg}")
    logging.info(f"Baostock login successful on {date}")

    # 获取所有股票代码
    rs = bs.query_all_stock(day=date)
    stock_list = []
    while (rs.error_code == "0") & rs.next():
        stock_info = rs.get_row_data()
        stock_code = stock_info[0]  # 第一列为股票代码
        if stock_code.startswith("sh.") or stock_code.startswith("sz."):
            stock_list.append({
                "code": stock_code,
                "name": stock_info[1],  # 股票名称
                "status": stock_info[2]  # 上市状态
            })

    # 登出 Baostock
    bs.logout()
    logging.info(f"Retrieved {len(stock_list)} A-share stock codes")

    # 转换为 DataFrame
    df = pd.DataFrame(stock_list)
    return df

def save_stock_codes(df, filename="a_stock_codes.csv"):
    """将股票代码保存到 CSV 文件"""
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    logging.info(f"Saved stock codes to {filename}")

if __name__ == "__main__":
    try:
        # 获取所有A股代码
        stock_df = get_all_a_stocks(date="2025-02-28")  # 可改为当前日期或其他日期
        print(f"Total A-share stocks retrieved: {len(stock_df)}")
        print(stock_df.head())  # 打印前几行查看

        # 保存到文件
        save_stock_codes(stock_df, "a_stock_codes.csv")
        print(f"Stock codes saved to a_stock_codes.csv")

    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
        print(f"Error: {e}")
