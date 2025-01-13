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

with open('C:\\Users\\12433\\Desktop\\data\\data\\sst2\\sst2.csv', 'rb') as f:#读取文件路径
    result = chardet.detect(f.read())
encoding = result['encoding']

if __name__ == "__main__":
    client = ZhipuAI(api_key="fa40780b56f34fa2ba2fa99f4f233ab2.pX1DXe60efQ6rVf9")
    # 读取实验数据
    file_path = 'C:\\Users\\12433\\Desktop\\data\\data\\sst2\\sst2.csv'
    data_list = read_csv(file_path=file_path)

    # 设置prompt
    prompt_dir = {'system_prompt': '你是一个善解人意的倾听者，接下来我会给你一系列文本，你要根据文本判断是正面情绪还是负面情绪,负面情绪一般有以下关键词：bake,bad,barely,action,bland;同时要根据语境判断',
                   'user_prompt': '请判断以下文本的情绪状态,负面情绪的话返回值为0；正面情绪返回值为1，不要有其他输出；如果没有输入就输出1，不要有除了0和1以外的输出:{{text}}'}
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
a = read_csv('C:\\Users\\12433\\Desktop\\data\\data\\sst2\\sst2.csv')
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