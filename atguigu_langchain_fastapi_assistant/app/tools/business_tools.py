import math
from datetime import datetime, timedelta

from langchain_core.tools import tool


# 作用：按城市名称查询内置天气字典；数据是演示用固定值，并非实时天气。
# 参数 city：中文城市名，如“北京”；必须匹配字典中的名称。
# 返回：天气说明字符串；不支持的城市返回提示，不抛出查询异常。
@tool
def get_weather(city: str) -> str:
    """查询课程案例中的模拟天气数据。

    Args:
        city: 城市名称，如“北京”“上海”“深圳”等。
    Returns:
        包含温度、天气状况、空气质量的模拟信息。
    """
    weather_db = {
        "北京": "多云，15-22℃，空气质量良，湿度 45%",
        "上海": "晴天，18-25℃，空气质量优，湿度 60%",
        "深圳": "小雨，22-28℃，空气质量优，湿度 75%",
        "成都": "阴天，16-23℃，空气质量良，湿度 70%",
        "杭州": "晴天，17-24℃，空气质量优，湿度 55%",
        "广州": "多云，21-29℃，空气质量良，湿度 72%",
    }
    result = weather_db.get(city)
    if result:
        return f"{city}：{result}"
    return (
        f"抱歉，暂不支持查询{city}的天气信息。"
        "当前支持：北京、上海、深圳、成都、杭州、广州"
    )


# 作用：执行数学表达式，并将计算结果或异常转换为文字。
# 参数 expression：表达式字符串，如 "sqrt(16) + 2 ** 3"；三角函数使用弧度。
# 返回：表达式及结果，或错误提示。限制内置函数不等于完整的安全沙箱。
@tool
def calculator(expression: str) -> str:
    """执行课程案例中的数学计算。

    Args:
        expression: 数学表达式。
    Returns:
        计算结果或错误信息。
    """
    try:
        safe_functions = {
            "sqrt": math.sqrt,
            "pow": pow,
            "abs": abs,
            "round": round,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "pi": math.pi,
            "e": math.e,
        }
        # 课程原案例采用受限 eval。学习项目保留该思路；不要直接作为生产环境任意表达式执行器。
        result = eval(expression, {"__builtins__": {}}, safe_functions)
        return f"{expression} = {result}"
    except Exception as exc:
        return (
            f"计算出错：{exc}\n"
            "提示：请检查表达式格式，支持的函数有 sqrt, abs, pow, sin, cos, tan, log"
        )


# 作用：读取运行机器的本地时间，不调用网络时间服务。
# 参数 query_type：默认 current（日期和时间）；date（日期）、tomorrow（明天）、
# yesterday（昨天）、weekday（星期）；不支持的值返回文字提示。
# 返回：格式化的时间或日期字符串。
@tool
def get_time_info(query_type: str = "current") -> str:
    """获取时间相关信息。

    Args:
        query_type: current/date/tomorrow/yesterday/weekday。
    Returns:
        时间信息字符串。
    """
    now = datetime.now()
    if query_type == "current":
        return now.strftime("当前时间：%Y年%m月%d日 %H:%M:%S")
    if query_type == "date":
        return now.strftime("今天是：%Y年%m月%d日")
    if query_type == "tomorrow":
        return (now + timedelta(days=1)).strftime("明天是：%Y年%m月%d日")
    if query_type == "yesterday":
        return (now - timedelta(days=1)).strftime("昨天是：%Y年%m月%d日")
    if query_type == "weekday":
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return f"今天是{weekdays[now.weekday()]}"
    return (
        f"不支持的查询类型：{query_type}。"
        "支持：current, date, tomorrow, yesterday, weekday"
    )


# 作用：利用固定汇率表，先将原币换算成人民币，再换算成目标币种。
# 参数 amount：源币种金额；from_curr：源币种代码；to_curr：目标币种代码。
# 币种支持 CNY/USD/EUR/GBP/JPY/HKD，代码会转为大写；汇率不实时更新。
# 返回：保留两位小数的换算说明，或不支持币种的提示。
@tool
def convert_currency(amount: float, from_curr: str, to_curr: str) -> str:
    """使用课程案例中的固定汇率表进行货币转换。

    Args:
        amount: 金额数值。
        from_curr: 源货币代码（CNY/USD/EUR/GBP/JPY/HKD）。
        to_curr: 目标货币代码（CNY/USD/EUR/GBP/JPY/HKD）。
    Returns:
        转换结果。
    """
    exchange_rates = {
        "CNY": 1.0,
        "USD": 0.14,
        "EUR": 0.13,
        "GBP": 0.11,
        "JPY": 20.8,
        "HKD": 1.09,
    }
    currency_names = {
        "CNY": "人民币",
        "USD": "美元",
        "EUR": "欧元",
        "GBP": "英镑",
        "JPY": "日元",
        "HKD": "港币",
    }

    from_curr = from_curr.upper()
    to_curr = to_curr.upper()
    if from_curr not in exchange_rates:
        return f"不支持的源货币：{from_curr}。支持：CNY, USD, EUR, GBP, JPY, HKD"
    if to_curr not in exchange_rates:
        return f"不支持的目标货币：{to_curr}。支持：CNY, USD, EUR, GBP, JPY, HKD"

    # 汇率表表示 1 元人民币可兑换的外币数量，因此先做除法还原人民币。
    cny_amount = amount / exchange_rates[from_curr]
    result_amount = cny_amount * exchange_rates[to_curr]
    return (
        f"{amount} {currency_names[from_curr]}（{from_curr}）= "
        f"{result_amount:.2f} {currency_names[to_curr]}（{to_curr}）"
    )


# 作用：在内置产品和新闻示例中做子串匹配，不会搜索互联网。
# 参数 keyword：关键词（英文区分大小写）；category：product 查产品、news 查新闻、
# all 同时查两类（默认）；其他类别不会匹配到结果。
# 返回：换行分隔的匹配文本，或未找到信息的提示。
@tool
def search_info(keyword: str, category: str = "all") -> str:
    """搜索课程案例中的模拟产品/新闻信息。

    Args:
        keyword: 搜索关键词。
        category: product/news/all。
    Returns:
        搜索结果。
    """
    products = {
        "手机": "iPhone 15 (¥5999), 小米14 (¥3999), 华为Mate60 (¥6999)",
        "笔记本": "MacBook Pro (¥12999), ThinkPad X1 (¥9999), 华为MateBook (¥7999)",
        "耳机": "AirPods Pro (¥1999), Sony WH-1000XM5 (¥2499)",
    }
    news = {
        "AI": "1. GPT-5 即将发布 2. AI 芯片市场增长 30% 3. 新AI法规出台",
        "科技": "1. 量子计算新突破 2. 6G 技术测试 3. 新能源汽车销量创新高",
    }

    results: list[str] = []
    if category in ["product", "all"]:
        for key, value in products.items():
            if keyword in key:
                results.append(f"【产品】{key}：{value}")
    if category in ["news", "all"]:
        for key, value in news.items():
            if keyword in key or keyword in value:
                results.append(f"【新闻】{key} 相关：{value}")

    return "\n".join(results) if results else f"未找到关于 '{keyword}' 的{category}信息"
