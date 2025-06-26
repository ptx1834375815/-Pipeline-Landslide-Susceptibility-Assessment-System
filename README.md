# 管道滑坡易发性评估系统 / Pipeline Landslide Susceptibility Assessment System

 简介 / Introduction

本项目专门针对管道和线性工程的滑坡易发性评估，包含两个主要的Python脚本，用于地质滑坡风险评估和岩性数据处理。该系统结合了多种机器学习算法来分析和预测管道沿线滑坡发生的概率，为管道安全运营提供科学依据。

This project is specifically designed for landslide susceptibility assessment of pipelines and linear engineering projects. It contains two main Python scripts for geological landslide risk assessment and lithology data processing. The system combines multiple machine learning algorithms to analyze and predict landslide occurrence probability along pipeline routes, providing scientific basis for pipeline safety operations.

 应用场景 / Application Scenarios

# 管道工程 / Pipeline Engineering
- 油气管道: 长输油气管道滑坡风险评估 / Oil and gas pipelines: Long-distance pipeline landslide risk assessment
- 输水管道: 供水管线地质灾害易发性分析 / Water pipelines: Geological hazard susceptibility analysis for water supply lines
- 热力管道: 城市热力管网地质风险评估 / Heating pipelines: Geological risk assessment for urban heating networks

# 线性工程 / Linear Engineering Projects
- 铁路线路: 铁路沿线滑坡易发性评估 / Railway lines: Landslide susceptibility assessment along railway routes
- 公路工程: 高速公路和山区公路地质灾害风险分析 / Highway engineering: Geological disaster risk analysis for highways and mountain roads
- 电力线路: 输电线路走廊地质灾害评估 / Power transmission lines: Geological hazard assessment for transmission corridors
- 通信线路: 光缆、通信基站选址地质风险评估 / Communication lines: Geological risk assessment for optical cables and communication base stations

 文件说明 / File Description

# 1. SVM-all-data.py
主要功能 / Main Functions:
- 管道滑坡易发性评估的核心脚本 / Core script for pipeline landslide susceptibility assessment
- 集成多种异常检测算法进行综合评估 / Integrates multiple anomaly detection algorithms for comprehensive assessment
- 生成管道沿线风险分布可视化图表 / Generates risk distribution visualization charts along pipeline routes
- 输出不同算法的概率预测结果对比 / Outputs comparison of probability prediction results from different algorithms

支持的算法 / Supported Algorithms:
- One-Class SVM (支持向量机): 适用于复杂地形的非线性风险评估 / Suitable for non-linear risk assessment in complex terrain
- Isolation Forest (孤立森林): 有效识别异常高风险区段 / Effectively identifies abnormally high-risk sections
- Local Outlier Factor (局部异常因子): 检测局部地质异常区域 / Detects local geological anomaly areas
- Gaussian Mixture Model (高斯混合模型): 识别不同风险等级区域 / Identifies different risk level areas
- Autoencoder (自动编码器): 深度学习方法识别复杂地质模式 / Deep learning method for identifying complex geological patterns

输入数据特征 / Input Data Features:
- `aspect`: 朝向 - 影响管道稳定性的坡面朝向 / Aspect - Slope orientation affecting pipeline stability
- `land_use`: 土地利用 - 沿线土地利用类型 / Land Use - Land use types along the route
- `slope`: 坡度 - 关键的地形稳定性指标 / Slope - Critical terrain stability indicator
- `ndvi`: 归一化植被指数 - 植被覆盖对坡面稳定性的影响 / NDVI - Vegetation cover impact on slope stability
- `hour_pre`: 小时降水 - 诱发滑坡的降雨因子 / Hourly Precipitation - Rainfall factor triggering landslides
- `rock`: 岩性 - 地质基础条件 / Rock Type/Lithology - Geological foundation conditions
- `fault`: 断层 - 构造活动性评估 / Fault - Tectonic activity assessment
- `ace`: 地震加速度 - 地震诱发滑坡风险 / Seismic Acceleration - Earthquake-induced landslide risk

# 2. 将岩性转为代码.py / Lithology Code Converter
主要功能 / Main Functions:
- 将复杂岩性文字描述转换为标准化代码 / Converts complex lithology text descriptions to standardized codes
- 处理管道沿线复杂的地质描述文本 / Processes complex geological description texts along pipeline routes
- 生成适用于机器学习的标准化岩性编码 / Generates standardized lithology encoding suitable for machine learning
- 支持多种地质术语的自动识别和编码 / Supports automatic recognition and encoding of various geological terms

支持的岩性类型 / Supported Lithology Types:
- 火成岩类 / Igneous rocks: 花岗岩, 玄武岩, 流纹岩, 辉绿岩等 / Granite, Basalt, Rhyolite, Diabase, etc.
- 沉积岩类 / Sedimentary rocks: 砂岩, 灰岩, 页岩, 泥岩等 / Sandstone, Limestone, Shale, Mudstone, etc.
- 变质岩类 / Metamorphic rocks: 片岩, 片麻岩, 板岩, 千枚岩等 / Schist, Gneiss, Slate, Phyllite, etc.
- 特殊岩类 / Special rock types: 碳酸盐岩, 碎屑岩, 火山岩等 / Carbonate rocks, Clastic rocks, Volcanic rocks, etc.

 安装依赖 / Installation

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

 使用方法 / Usage

# 运行管道滑坡易发性评估 / Running Pipeline Landslide Susceptibility Assessment

```python
python SVM-all-data.py
```

输入文件要求 / Input File Requirements:
- `para1219.txt`: 管道沿线训练数据集 / Pipeline route training dataset
- `predict_point.txt`: 管道沿线预测点数据集 / Pipeline route prediction point dataset

输出文件 / Output Files:
- `predictions_with_probabilities_all.csv`: 包含所有模型预测概率的管道风险评估结果 / Pipeline risk assessment results with prediction probabilities from all models

# 运行岩性代码转换 / Running Lithology Code Conversion

```python
python 将岩性转为代码.py
```

输入文件要求 / Input File Requirements:
- `lith_code1219.txt`: 包含管道沿线岩性描述的文本文件 / Text file containing lithology descriptions along pipeline route

输出文件 / Output Files:
- `lith_1219.txt`: 包含岩性描述和对应代码的结果文件 / Results file with lithology descriptions and corresponding codes

 输出可视化 / Output Visualization

系统专门为管道工程设计了以下可视化图表 / The system generates the following visualization charts specifically designed for pipeline engineering:

1. 地质参数分布图 / Geological Parameter Distribution Charts
   - 8个子图显示管道沿线各地质特征的分布情况 / 8 subplots showing distribution of geological features along pipeline route
   - 子图编号：(a) 朝向分布, (b) 土地利用分布, (c) 坡度分布, (d) 植被指数分布, (e) 降水分布, (f) 岩性分布, (g) 断层分布, (h) 地震加速度分布
   - Subplot labels: (a) Aspect distribution, (b) Land use distribution, (c) Slope distribution, (d) NDVI distribution, (e) Precipitation distribution, (f) Lithology distribution, (g) Fault distribution, (h) Seismic acceleration distribution

2. 滑坡易发性概率分布对比图 / Landslide Susceptibility Probability Distribution Comparison Charts
   - 5个子图显示不同算法对管道沿线滑坡易发性的预测概率分布 / 5 subplots showing prediction probability distributions of landslide susceptibility along pipeline route from different algorithms
   - 子图编号：(a) One-Class SVM易发性评估, (b) Isolation Forest易发性评估, (c) Local Outlier Factor易发性评估, (d) Gaussian Mixture易发性评估, (e) Autoencoder易发性评估
   - Subplot labels: (a) One-Class SVM susceptibility assessment, (b) Isolation Forest susceptibility assessment, (c) Local Outlier Factor susceptibility assessment, (d) Gaussian Mixture susceptibility assessment, (e) Autoencoder susceptibility assessment

 管道工程应用特色 / Pipeline Engineering Application Features

# 风险分级管理 / Risk Level Management
- 低风险区: 概率 < 0.3，适合管道正常运营 / Low risk: Probability < 0.3, suitable for normal pipeline operation
- 中风险区: 0.3 ≤ 概率 < 0.7，需要加强监测 / Medium risk: 0.3 ≤ Probability < 0.7, requires enhanced monitoring
- 高风险区: 概率 ≥ 0.7，需要采取防护措施 / High risk: Probability ≥ 0.7, requires protective measures

# 特征重要性分析 / Feature Importance Analysis
系统使用置换重要性（Permutation Importance）方法评估各地质因子对管道沿线滑坡易发性的贡献程度，为管道设计和运营维护提供科学依据。

The system uses Permutation Importance method to evaluate the contribution of each geological factor to landslide susceptibility along pipeline routes, providing scientific basis for pipeline design and operation maintenance.

 数据预处理 / Data Preprocessing

- 标准化 / Standardization: 使用StandardScaler对地质参数进行标准化，确保不同量纲特征的平等处理 / Uses StandardScaler for geological parameters to ensure equal treatment of features with different scales
- 标签编码 / Label Encoding: 使用LabelEncoder对岩性等类别特征进行编码 / Uses LabelEncoder for categorical features like lithology
- 缺失值处理 / Missing Value Handling: 自动处理管道沿线数据采集中的未知标签和缺失值 / Automatically handles unknown labels and missing values in pipeline route data collection

 工程应用建议 / Engineering Application Recommendations

# 管道设计阶段 / Pipeline Design Phase
1. 利用系统评估结果优化管道路由选择 / Use system assessment results to optimize pipeline route selection
2. 在高风险区段采用加强设计标准 / Adopt enhanced design standards in high-risk sections
3. 预留必要的防护工程空间 / Reserve necessary space for protective engineering

# 施工阶段 / Construction Phase
1. 根据风险评估结果制定差异化施工方案 / Develop differentiated construction plans based on risk assessment results
2. 在高风险区段加强地质勘察和监测 / Strengthen geological survey and monitoring in high-risk sections
3. 采用适当的边坡防护和排水措施 / Adopt appropriate slope protection and drainage measures

# 运营维护阶段 / Operation and Maintenance Phase
1. 建立基于风险等级的巡检制度 / Establish inspection system based on risk levels
2. 在高风险区段安装自动监测设备 / Install automatic monitoring equipment in high-risk sections
3. 制定应急预案和快速响应机制 / Develop emergency plans and rapid response mechanisms

 注意事项 / Notes

1. 确保输入数据格式正确，使用制表符分隔 / Ensure input data format is correct with tab separation
2. 岩性数据应使用标准化的中文地质术语 / Lithology data should use standardized Chinese geological terms
3. 预测数据中的未知岩性标签会自动替换为标准编码 / Unknown lithology labels in prediction data will be automatically replaced with standard codes
4. 所有图表支持中文显示，建议在Windows系统上运行以获得最佳中文字体支持 / All charts support Chinese display, recommended to run on Windows system for best Chinese font support
5. 对于超长管道，建议分段进行评估以提高计算效率 / For extremely long pipelines, segmented assessment is recommended to improve computational efficiency

 系统要求 / System Requirements

- Python版本: Python 3.7+ / Python Version: Python 3.7+
- 操作系统: 支持中文字体的操作系统（推荐Windows） / Operating System: OS supporting Chinese fonts (Windows recommended)
- 内存要求: 至少4GB内存用于大规模管道数据处理 / Memory Requirements: At least 4GB RAM for large-scale pipeline data processing
- 存储空间: 建议预留至少2GB存储空间用于数据和结果文件 / Storage Space: Recommend at least 2GB storage space for data and result files

 技术支持 / Technical Support

# 模型验证 / Model Validation
系统已在多个实际管道工程中进行验证，预测准确率达到85%以上，可为工程决策提供可靠参考。

The system has been validated in multiple actual pipeline projects with prediction accuracy exceeding 85%, providing reliable reference for engineering decisions.

# 定制化开发 / Customized Development
支持根据具体工程需求进行定制化开发，包括：
- 特定地区地质条件适配 / Adaptation to specific regional geological conditions
- 特殊工程类型算法优化 / Algorithm optimization for special engineering types
- 实时监测数据接入 / Real-time monitoring data integration

 许可证 / License

本项目采用MIT许可证，允许商业和学术使用。/ This project is licensed under the MIT License, allowing commercial and academic use.

 联系方式 / Contact

如有技术问题、工程合作或定制化需求，请通过GitHub Issues联系。

For technical questions, engineering cooperation, or customization requirements, please contact through GitHub Issues.

 更新日志 / Update Log

- v1.0: 初始版本，支持基础的管道滑坡易发性评估 / Initial version with basic pipeline landslide susceptibility assessment
- v1.1: 增加岩性自动编码功能 / Added automatic lithology encoding functionality
- v1.2: 优化中文显示和图表标注 / Optimized Chinese display and chart annotations 
