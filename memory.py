# 作用：保留全部 system 消息，加上最近 max_pairs * 2 条普通消息。
# 参数 messages：消息字典列表，每条通常含 role（角色）与 content（内容）。
# 参数 max_pairs：窗口轮数，默认 3；调用处可传配置值，应使用正整数。
# 返回：新列表，不修改原列表，也不复制内部字典；不会清理长期内存历史。
# 注意：按条数截取，不保证窗口起点为 user；max_pairs=0 时切片会保留全部消息。
def keep_recent_messages(messages, max_pairs=3):
    """
    保留最近的 N 轮对话。

    每轮 = user + assistant；system 消息始终保留。
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    conversation_msgs = [m for m in messages if m.get("role") != "system"]
    recent_msgs = conversation_msgs[-(max_pairs * 2):]
    return system_msgs + recent_msgs
