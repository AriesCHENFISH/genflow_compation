# 请安装 OpenAI SDK : pip install openai
# apiKey 获取地址： https://console.bce.baidu.com/qianfan/ais/console/apiKey
# 支持的模型列表： https://cloud.baidu.com/doc/qianfan-docs/s/7m95lyy43

from openai import OpenAI
client = OpenAI(
    base_url='https://qianfan.baidubce.com/v2',
api_key='bce-v3/ALTAK-cr1JliXkzzpKeAKi5SWFC/b61e551c22fe3160b54706f6f46dfd5248c3cb24'
)
def get_model_response(robot_prompt, user_content="hello", model="qwen3.5-27b"):
    """
    调用千帆API获取模型响应
    
    Args:
        robot_prompt: 系统提示词
        user_content: 用户输入内容
        model: 模型名称
    
    Returns:
        dict: 包含 result（主要回复）、reasoning（推理过程）、usage（token统计）的字典
    """
    messages = [
        {"role": "system", "content": robot_prompt},
        {"role": "user", "content": user_content}
    ]
    response = client.chat.completions.create(
        model=model, 
        messages=messages, 
        temperature=0.6, 
        top_p=0.95,
        extra_body={ 
            "stop":[], 
            "enable_thinking": False
        },
        max_tokens=65536
    )
    
    # 解析响应
    result = {
        "model": response.model or model,  # 返回的模型名称
        "result": "",
        "reasoning": "",
        "usage": {}
    }
    
    if response.choices and len(response.choices) > 0:
        choice = response.choices[0]
        # 获取主要回复内容
        result["result"] = choice.message.content or ""
        # 获取推理过程（如果有）
        result["reasoning"] = getattr(choice.message, 'reasoning_content', "") or ""
    
    # 获取token使用统计
    if response.usage:
        result["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "reasoning_tokens": getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 0) if response.usage.completion_tokens_details else 0
        }
    
    return result


if __name__ == '__main__':
    result = get_model_response("你是一个有帮助的助手", "你好")
    print("模型名称:", result["model"])
    print("主要回复:", result["result"])
    # print("推理过程:", result["reasoning"])
    print("Token统计:", result["usage"])