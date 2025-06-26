import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import matplotlib as mpl
from matplotlib import font_manager
# import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# 设置图形风格
sns.set_style("whitegrid")
sns.set_palette("deep")

# 将图幅宽度设为165mm，并转换为英寸
mm_to_inch = 1/25.4
fig_width = 165 * mm_to_inch
# 设置全局字体大小为14磅
mpl.rcParams['font.size'] = 14

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 支持中文显示
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取主数据集
df = pd.read_csv('D:\\data\\GEE\\para1219.txt', delimiter='\t', encoding='utf-8')

# 获取训练数据中'rock'列的所有标签值
known_labels = df['rock'].unique()

# 选择特征
features = ['aspect', 'land_use', 'slope', 'ndvi', 'hour_pre', 'rock', 'fault', 'ace']
# 添加中英文映射字典
chinese_names = {'aspect': '朝向', 'land_use': '土地利用', 'slope': '坡度', 'ndvi': '归一化植被指数', 'hour_pre': '小时降水', 'rock': '岩性', 'fault': '断层', 'ace': '地震加速度'}

# 岩性代码到中文的映射字典
lithology_code_to_chinese = {
    "A": "花岗岩", "B": "正长岩", "C": "闪长岩", "D": "玄武岩", "E": "石英岩",
    "F": "大理岩", "G": "板岩", "H": "片岩", "I": "片麻岩", "J": "千枚岩",
    "K": "凝灰岩", "L": "角砾岩", "M": "砾岩", "N": "砂岩", "O": "粉砂岩",
    "P": "灰岩", "Q": "白云岩", "R": "页岩", "S": "泥岩", "T": "斑岩",
    "U": "辉绿岩", "V": "流纹岩", "W": "土", "X": "碳酸盐岩", "Y": "碎屑岩",
    "Z": "变质沉积岩", "a": "变质火山岩", "b": "赤铁矿", "c": "硅质岩", "d": "火山岩",
    "e": "生物屑灰岩", "f": "鲕状灰岩", "g": "夹灰岩", "h": "夹页岩", "i": "夹煤",
    "j": "变泥砂质岩", "k": "基性火山岩", "l": "夹碳酸盐岩"
}

x = df[features]

# 处理类别变量并进行编码
label_encoder = LabelEncoder()
for col in ['rock']:
    x[col] = label_encoder.fit_transform(x[col])

# 标准化数据
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# 绘制参数分布
plt.figure(figsize=(fig_width, fig_width * 12 / 15))

# 子图编号列表
subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

for i, feature in enumerate(features):
    plt.subplot(2, 4, i + 1)

    # 对于 'rock' 和其他类别特征使用 countplot
    if feature == 'rock':
        # 创建一个临时的数据框，将岩性代码转换为中文显示
        rock_chinese = df[feature].map(lithology_code_to_chinese).fillna(df[feature])
        sns.countplot(x=rock_chinese, palette="Set2")
        plt.xticks(rotation=45, ha='right')  # 旋转x轴标签以便更好显示中文
    else:
        sns.histplot(df[feature], kde=False, bins=30, stat="count")  # 数量统计分布图

    plt.title(f"({subplot_labels[i]}) {chinese_names[feature]}分布")
    plt.xlabel(f"{chinese_names[feature]}")
    plt.ylabel('数量')

plt.tight_layout()
plt.show()

# 读取预测数据
predict_data = pd.read_csv('D:\\data\\GEE\\predict_point.txt', delimiter='\t', encoding='utf-8')

# 处理类别变量并进行编码
label_encoder = LabelEncoder()

# 获取 'rock' 列中的所有预测标签
predict_data_labels = predict_data['rock'].unique()

# 找到不在训练标签中的标签（即未知标签）
unknown_labels = [label for label in predict_data_labels if label not in known_labels]

# 替换未知标签为已知标签（例如 'N'）
for label in unknown_labels:
    predict_data['rock'] = predict_data['rock'].replace(label, 'N')  # 将所有未知标签替换为 'N'

# 对 'rock' 列进行标签编码
label_encoder.fit(df['rock'])  # 只在训练数据上拟合
predict_data['rock'] = label_encoder.transform(predict_data['rock'])  # 使用训练数据上的编码器进行转换

# 标准化预测数据
predict_data_scaled = scaler.transform(predict_data[features])


# 定义模型
models = {
    "One-Class SVM": OneClassSVM(nu=0.1, kernel="rbf", gamma='auto'),
    "Isolation Forest": IsolationForest(contamination=0.1, random_state=42),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True),
    "Gaussian Mixture": GaussianMixture(n_components=2, covariance_type='full', random_state=42)
}

# 训练模型
for name, model in models.items():
    if name == "Gaussian Mixture":
        model.fit(x_scaled)
        # GMM使用对数概率作为异常分数
        predict_data[f'{name}_probability'] = np.exp(model.score_samples(predict_data_scaled))
    else:
        model.fit(x_scaled)
        score = model.decision_function(predict_data_scaled) if hasattr(model, "decision_function") else -model.score_samples(predict_data_scaled)
        predict_data[f'{name}_probability'] = 1 / (1 + np.exp(-score))

# 训练模型并输出特征重要性
for name, model in models.items():
    print(f"Training {name} model...")

    # 对于每个模型进行训练
    if name == "Gaussian Mixture":
        model.fit(x_scaled)
        # GMM使用对数概率作为异常分数
        predict_data[f'{name}_probability'] = np.exp(model.score_samples(predict_data_scaled))

        # GaussianMixture没有直接的特征重要性，使用permutation importance估算
        result = permutation_importance(model, x_scaled, np.ones(x_scaled.shape[0]), n_repeats=10, random_state=42)
        print(f"Feature importance (Permutation Importance) for {name}:")
        for i, feature in enumerate(features):
            print(f"{feature}: {result.importances_mean[i]:.4f}")

    else:
        model.fit(x_scaled)
        score = model.decision_function(predict_data_scaled) if hasattr(model,
                                                                        "decision_function") else -model.score_samples(
            predict_data_scaled)
        predict_data[f'{name}_probability'] = 1 / (1 + np.exp(-score))

        # 对于其他模型，使用permutation importance
        result = permutation_importance(model, x_scaled, np.ones(x_scaled.shape[0]), n_repeats=10, random_state=42,
                                        scoring='accuracy')
        print(f"Feature importance (Permutation Importance) for {name}:")
        for i, feature in enumerate(features):
            print(f"{feature}: {result.importances_mean[i]:.4f}")

# 自动编码器
input_dim = x_scaled.shape[1]
encoding_dim = 32

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')
autoencoder.fit(x_scaled, x_scaled, epochs=50, batch_size=256, shuffle=True, validation_split=0.2, verbose=0)

encoded_imgs = encoder.predict(predict_data_scaled)
decoded_imgs = autoencoder.predict(predict_data_scaled)
reconstruction_error = np.mean(np.square(predict_data_scaled - decoded_imgs), axis=1)
predict_data['Autoencoder_probability'] = 1 - reconstruction_error  # 转换为概率形式

# 绘制概率分布图进行对比
plt.figure(figsize=(fig_width, fig_width * 20 / 15))

# 概率分布图的子图编号
prob_subplot_labels = ['a', 'b', 'c', 'd', 'e']

for i, (name, _) in enumerate(models.items()):
    plt.subplot(5, 1, i+1)
    sns.histplot(predict_data[f'{name}_probability'], kde=True, bins=30, stat="density")
    plt.title(f"({prob_subplot_labels[i]}) {name}概率分布")
    plt.xlabel('滑坡概率')
    plt.ylabel('密度')

# 单独为自动编码器绘制图
plt.subplot(5, 1, 5)
sns.histplot(predict_data['Autoencoder_probability'], kde=True, bins=30, stat="density")
plt.title('(e) 自动编码器概率分布')
plt.xlabel('滑坡概率')
plt.ylabel('密度')

plt.tight_layout()
plt.show()

# 保存预测结果
predict_data.to_csv('D:\\data\\GEE\\predictions_with_probabilities_all.csv', sep='\t', index=False, encoding='utf-8')
print("Prediction results with probabilities from all models saved.")

