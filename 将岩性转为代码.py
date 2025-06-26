import re

lithology_dict = {
    "花岗岩": "A", "正长岩": "B", "闪长岩": "C", "玄武岩": "D", "石英岩": "E",
    "大理岩": "F", "板岩": "G", "片岩": "H", "片麻岩": "I", "千枚岩": "J",
    "凝灰岩": "K", "角砾岩": "L", "砾岩": "M", "砂岩": "N", "粉砂岩": "O",
    "灰岩": "P", "白云岩": "Q", "页岩": "R", "泥岩": "S", "斑岩": "T",
    "辉绿岩": "U", "流纹岩": "V", "土": "W", "碳酸盐岩": "X", "碎屑岩": "Y",
    "变质沉积岩": "Z", "变质火山岩": "a", "赤铁矿": "b", "硅质岩": "c", "火山岩": "d",
    "生物屑灰岩": "e", "鲕状灰岩": "f", "夹灰岩": "g", "夹页岩": "h", "夹煤": "i",
    "变泥砂质岩": "j", "基性火山岩": "k", "夹碳酸盐岩": "l"
}


# 从txt文件中读取岩性描述数据
def read_lithology_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lithology_data = file.readlines()
    return [line.strip() for line in lithology_data]


# 提取岩性代号的函数
def extract_lithology_codes(description):
    # 找到可能的岩性描述并匹配
    lithologies_found = []

    # 优先匹配岩性字典中的岩性
    for lithology_name in lithology_dict:
        if lithology_name in description:
            lithologies_found.append(lithology_dict[lithology_name])

    # 如果没有找到匹配的岩性，检查夹层和组合
    if not lithologies_found:
        matches = re.findall(r'([\u4e00-\u9fa5]+岩)', description)
        for match in matches:
            lithology_code = lithology_dict.get(match, "")
            if lithology_code:
                lithologies_found.append(lithology_code)

    # 如果描述包含“土”字，可以专门处理
    if "土" in description:
        lithologies_found.append("W")  # “土”字的代号

    # 确保输出最多3个代号
    lithologies_found = lithologies_found[:3]

    # 返回找到的代号，如果没有匹配，返回“未找到匹配的岩性”
    return "".join(lithologies_found) if lithologies_found else "未找到匹配的岩性"


# 读取输入文件并处理数据
def process_lithology_file(input_file, output_file):
    lithology_data = read_lithology_from_txt(input_file)
    results = []

    for description in lithology_data:
        code = extract_lithology_codes(description)
        results.append((description, code))

    # 保存结果到输出文件，分成两列
    with open(output_file, 'w', encoding='utf-8') as output:
        output.write("描述\t代号\n")  # 表头
        for description, code in results:
            output.write(f"{description}\t{code}\n")

    print(f"处理完成，结果已保存到 {output_file}")


# 输入文件路径和输出文件路径
input_file = "D:\data\GEE\lith_code1219.txt"  # 替换为你的输入文件路径
output_file = "lith_1219.txt"  # 输出文件路径

# 调用函数处理数据
process_lithology_file(input_file, output_file)


