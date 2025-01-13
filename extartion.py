import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import re
import chardet

# 检测文件编码
with open('*******************************', 'rb') as f:#读取文件路径
    result = chardet.detect(f.read())
encoding = result['encoding']

# 1. 加载CSV文件
def load_data(file_path):
 try:
     data = pd.read_csv(file_path,encoding=encoding)
     return data
 except FileNotFoundError:
     print(f"文件 {file_path} 未找到，请检查路径。")
     return None

file_path = '*******************************'#存贮文件路径
data = load_data(file_path)

if data is not None:
 # 检查必要的列名是否存在
 required_columns = ['sentence', 'label']
 missing_columns = [col for col in required_columns if col not in data.columns]
 if missing_columns:
     print(f"错误：DataFrame中缺少必要的列：{missing_columns}")
 else:
     texts = data['sentence']
     labels = data['label']

     # 处理 NaN 值
     texts = texts.fillna('')

     # 确保所有值都是字符串类型
     texts = texts.astype(str)

     # 2. 文本预处理
     def preprocess_text(text):
         # 去除标点符号和数字
         text = re.sub(r'[^\w\s]', '', text)
         text = re.sub(r'\d+', '', text)
         # 转换为小写
         return text.lower()

     texts = texts.apply(preprocess_text)

     # 3. 特征提取
     # 词袋模型
     bow_vectorizer = CountVectorizer(max_features=1000)
     bow_features = bow_vectorizer.fit_transform(texts)

     # TF-IDF
     tfidf_vectorizer = TfidfVectorizer(max_features=1000)
     tfidf_features = tfidf_vectorizer.fit_transform(texts)


     # 4. 添加统计特征
     def extract_statistical_features(text):
         length = len(text)
         num_words = len(text.split())
         num_unique_words = len(set(text.split()))
         return [length, num_words, num_unique_words]


     statistical_features = np.array([extract_statistical_features(text) for text in texts])

     # 5. 合并特征
     combined_features = hstack([bow_features, tfidf_features, statistical_features])

     # 6. 保存特征到新的CSV文件
     feature_names = bow_vectorizer.get_feature_names_out().tolist() + \
                     tfidf_vectorizer.get_feature_names_out().tolist() + \
                     ['length', 'num_words', 'num_unique_words']

     features_df = pd.DataFrame(combined_features.toarray(), columns=feature_names)
     features_df['label'] = labels
     output_path = '********************************'
     features_df.to_csv(output_path, index=False)
     print(f"特征提取完成，结果已保存到 {output_path}")

     # 7. 查看和可视化特征
     # 打印部分特征矩阵
     print(features_df.head())

     # 可视化特征重要性（假设有一个分类任务）
     model = LogisticRegression()
     model.fit(bow_features, labels)
     feature_importance = model.coef_[0]

     plt.figure(figsize=(10, 6))
     plt.bar(feature_names[:60], feature_importance[:60])  # 显示前40个特征
     plt.xlabel('Features')
     plt.ylabel('Importance')
     plt.xticks(rotation=90)
     plt.title('Top 60 Feature Importance')
     plt.show()

     # 可视化统计特征分布
     statistical_df = pd.DataFrame(statistical_features, columns=['Length', 'Num Words', 'Num Unique Words'])
     sns.pairplot(statistical_df)
     plt.show()
else:
    print("数据加载失败，请检查文件路径。")
