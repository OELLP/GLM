import pandas as pd
from zhipuai import ZhipuAI
import chardet

def read_parquet(file_path: str) -> list:
    df = pd.read_parquet(file_path)
    df_list = df.values.tolist()
    return df_list


def read_csv(file_path: str) -> list:
    df = pd.read_csv(file_path,encoding=encoding)
    df_list = df.values.tolist()
    return df_list

with open('****************', 'rb') as f:#读取文件路径
    result = chardet.detect(f.read())
encoding = result['encoding']

if __name__ == "__main__":
    client = ZhipuAI(api_key="****************")#输入API
    # 读取实验数据
    file_path = '****************'#文件路径
    data_list = read_csv(file_path=file_path)

    # 设置prompt
    prompt_dir = {'system_prompt': '你是老板秘书，你的任务是帮助老板辨别垃圾邮件,垃圾邮件一般为广告推销，垃圾邮件一般含有下划线以及以下特征：商业广告性质明显，虚假信息或夸大其词，包含恶意链接或附件，内容格式混乱或者以下关键词：about,anything;相反的，如果有以下关键词，一般为非垃圾邮件：anywhere,america,age,African,adult,accept。当然，不能只根据关键词判断，要根据语境判断垃圾邮件',
                   'user_prompt': '请判断以下文本是否为垃圾邮件,是的话返回值为1否则为0，不要有其他输出；如果没有输入就输出1，不要有除了0和1以外的输出:{{text}}'}
    with open('data.txt', 'w') as file:
        for i, row in enumerate(data_list):
            # 生成用户提示
            system_prompt = prompt_dir['system_prompt']
            user_prompt = prompt_dir['user_prompt'].replace('{{text}}', str(row[0]))


            # 整合prompt并进行大模型推理
            response = client.chat.completions.create(
                model="glm-4-plus",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )

            # 获取结果并写入文件
            result = response.choices[0].message.content
            file.write(result + '\n')
            print(f"{result}")
a = read_csv('**************')#文件路径
b = [row[1] for row in a]
c = []
with open('data.txt','r') as file:
    for line in file:
        num = int(line.strip())
        c.append(num)
count = 0
for i in range(len(b)):
    if b[i] != c[i]:
        count += 1
count = float(count)
len = float(len(b))
rate = 1 - count / len
print(rate)
