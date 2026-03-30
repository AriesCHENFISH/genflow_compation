
import os
import random
import requests
import json
import time
import traceback

# 千帆API配置
QIANFAN_BASE_URL = 'https://qianfan.baidubce.com/v2'
QIANFAN_API_KEY = 'bce-v3/ALTAK-cr1JliXkzzpKeAKi5SWFC/b61e551c22fe3160b54706f6f46dfd5248c3cb24'


def get_bns_server(bnsname: str):
        try:
            fout = os.popen("get_instance_by_service -a %s" % bnsname)
            res = []
            for x in fout.readlines():
                mid = x.split(" ")
                if len(mid) >= 4:
                    res.append([mid[1], mid[3]])
            return res
        except Exception:
            return []


def invoke_model_qianfan(user_prompt, system_prompt=None, model="qwen3.5-27b",
                         temperature=0.6, max_tokens=65536, times=3, timeout=200):
    """
    使用千帆API调用模型

    Args:
        user_prompt: 用户提示词
        system_prompt: 系统提示词
        model: 模型名称，如 qwen3.5-27b
        temperature: 温度参数
        max_tokens: 最大生成token数
        times: 重试次数
        timeout: 超时时间（秒）

    Returns:
        str: 模型回复内容，或 None/错误码
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("[错误] 请安装 OpenAI SDK: pip install openai")
        return None

    client = OpenAI(
        base_url=QIANFAN_BASE_URL,
        api_key=QIANFAN_API_KEY
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    trytime = 0
    while trytime < times:
        try:
            print(f"[千帆API] 第 {trytime + 1}/{times} 次尝试... 模型: {model}")

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=0.95,
                extra_body={
                    "stop": [],
                    "enable_thinking": False
                },
                max_tokens=max_tokens,
                timeout=timeout
            )

            # 解析响应
            if response.choices and len(response.choices) > 0:
                result = response.choices[0].message.content or ""
                print(f"[千帆API] 成功返回，长度: {len(result)} 字符")
                return result
            else:
                print(f"[千帆API] 返回为空")
                trytime += 1
                time.sleep(2)

        except Exception as e:
            print(f"[千帆API] 异常: {str(e)}")
            trytime += 1
            time.sleep(2)
            if trytime == times:
                print(f"[千帆API] 重试次数达到上限")
                return None

    return None


def invoke_model_ds_rr(user_prompt, system_prompt=None, user_id=None, chat_id=None,
                           query_id=None, channel=None, max_tokens=12000, temperature=0.95,
                           times=5, timeout=200, model="ali-kimi-k2.5"):
        """
        模型调用入口函数。
        根据模型名称自动选择调用方式：
        - qwen3.5-27b 及相关 qwen 模型: 使用千帆API
        - 其他模型: 使用内部BNS服务
        """
        # 判断是否使用千帆API（千帆支持的模型）
        qianfan_models = ["qwen3.5-27b", "qwen-turbo", "qwen-plus", "qwen-max",
                         "qwen2.5-72b-instruct", "qwen2.5-14b-instruct",
                         "qwen2-72b-instruct", "deepseek-v3", "deepseek-v3.2"]

        # 如果是千帆支持的模型，使用千帆API
        if any(qwen_model in model.lower() for qwen_model in qianfan_models):
            return invoke_model_qianfan(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                times=times,
                timeout=timeout
            )

        # 否则使用原有内部BNS服务
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        user_id = user_id or "6572030879"
        chat_id = chat_id or 111110
        query_id = query_id or 111110
        channel = channel or "wenku_all_survey_offline"
        NEW_WENKU_YIYAN_BNS = "group.gdp-wenchainllm-online.pandora.all"

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer bce-v3/ALTAK-wCK6Ip8TSos16E8UJeRCa/3a0dbfac888de9e1c1b42064d899b49c297159d0",
        }
        api = "/wenchain/strategy/incommonusesse"

        ip_list = get_bns_server(bnsname=NEW_WENKU_YIYAN_BNS)
        if not ip_list:
            ip_list = get_bns_server(bnsname=NEW_WENKU_YIYAN_BNS)
        if not ip_list:
            print("[错误] 无法获取 BNS 实例")
            return 402

        trytime = 0
        while trytime < times:
            try:
                ip, port = random.choice(ip_list)
                if [ip, port] in ip_list and len(ip_list) > 1:
                    ip_list.remove([ip, port])

                url = "http://{ip}:{port}{api}".format(ip=ip, port=port, api=api)
                payload = {
                    "messages": messages,
                    "user_id": user_id,
                    "channel": channel,
                    "chat_id": int(chat_id),
                    "query_id": int(query_id),
                    "model": model,
                    "stream": True,
                    "message_user": messages,
                    "message_prompt": messages,
                }

                print(f"[LLM请求] 第 {trytime + 1}/{times} 次尝试...")
                response = requests.post(url, json=payload, headers=headers, stream=True, timeout=timeout)

                if response.status_code == 200:
                    res_str = ""
                    model_change = False
                    model_low = False

                    for line in response.iter_lines():
                        if not line:
                            continue
                        decoded_line = line.decode("utf-8")

                        if 'data: [DONE]' in decoded_line:
                            break

                        if decoded_line.startswith("data:"):
                            json_str = decoded_line[5:].strip()
                            try:
                                data_json = json.loads(json_str)

                                if model == "ali-glm-4.7" and data_json.get("model", "") != "ali-glm-4.7":
                                    print(f"[警告] 模型发生降级! 当前返回模型: {data_json.get('model')}")
                                    if trytime + 1 >= times:
                                        if not model_low:
                                            print("[警告] 超过重试次数限制，接受降级结果。")
                                            model_low = True
                                    else:
                                        print("[重试] 触发降级重试机制，等待后重试...")
                                        model_change = True
                                        break

                                if data_json.get('need_clear_history') or data_json.get('finish_reason') == "content_filter":
                                    print(f"[风控] 模型输出命中风控: {json_str}")
                                    return 400

                                res_str += data_json.get('result', '')

                            except json.JSONDecodeError:
                                continue

                    if model_change:
                        time.sleep(random.randint(5, 10))
                        trytime += 1
                        continue

                    return res_str
                else:
                    print(f"[错误] HTTP {response.status_code}")
                    trytime += 1
                    time.sleep(2)
                    if trytime == times:
                        print("[错误] 重试次数达到上限")
                        return response.status_code

            except Exception as e:
                time.sleep(2)
                print(f"[异常] {e}\n{traceback.format_exc()}")
                trytime += 1

        return 401
