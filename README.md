# README

author: 快刀切草莓君
mail: li_zaaachary@163.com

## 文件结构

- Data：样例数据
- CODE: 代码文件
- Shell: 训练和预测的脚本文件
- Model：训练好的模型

## 预测

- `pip install -r requirement.txt`
- 修改 Shell 中 `infer.sh` 的路径 (PTM_name_or_path, model_path, input_path, output_path)
- `cd Shell`
- `sh infer.sh`