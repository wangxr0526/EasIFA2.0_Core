# EasIFA 批量预测指南

## 批量预测功能

EasIFA 支持通过 JSON 文件进行批量预测，可以一次处理多个蛋白质。

## JSON 输入格式

批量输入文件必须是一个 JSON 数组，每个元素代表一个蛋白质的预测任务。

### 必需字段

- `id` (string): 蛋白质的唯一标识符

### 可选字段（至少提供其中一个）

- `enzyme_structure` (string): PDB 文件路径
- `enzyme_sequence` (string): 氨基酸序列（单字母代码）
- `rxn_smiles` (string): 反应 SMILES 字符串（格式：reactant>>product）

## 示例 JSON 文件

```json
[
  {
    "id": "protein_1",
    "enzyme_structure": "path/to/protein1.pdb",
    "rxn_smiles": "O.OCC1OC(OC2C(O)C(CO)OC(OC3C(O)C(O)OC(CO)C3O)C2O)C(O)C(O)C1O>>O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO"
  },
  {
    "id": "protein_2",
    "enzyme_sequence": "MSPRPLRALLGAAAAALVSAAALAFPSQAAANDSPFYVNPNMSS...",
    "rxn_smiles": "CC(=O)O>>CCO"
  },
  {
    "id": "protein_3_no_reaction",
    "enzyme_sequence": "MSPRLKQVNLCDEFGHIKLMNPQRSTVWY"
  },
  {
    "id": "protein_4_structure_only",
    "enzyme_structure": "path/to/protein4.pdb"
  }
]
```

## 使用方法

### 基本用法

```bash
python easifa_predict.py --batch-input batch_input.json --output batch_results.json
```

### 带详细输出

```bash
python easifa_predict.py --batch-input batch_input.json --output batch_results.json --verbose --pretty
```

### 指定 GPU

```bash
python easifa_predict.py --batch-input batch_input.json --output batch_results.json --gpu-id 0
```

### 只加载特定模型（节省内存）

```bash
python easifa_predict.py --batch-input batch_input.json --output batch_results.json --model-to-use wo_rxn_structures
```

## 输出格式

批量预测的输出 JSON 格式如下：

```json
{
  "batch_results": [
    {
      "id": "protein_1",
      "model_used": "all_features",
      "input": {
        "enzyme_structure": "path/to/protein1.pdb",
        "enzyme_sequence": "MSPRPLRALL...",
        "rxn_smiles": "reactant>>product"
      },
      "predictions": {
        "labels": [0, 0, 1, 2, 0, ...],
        "probabilities": [[0.9, 0.05, 0.03, 0.02], ...]
      },
      "sequence_length": 350,
      "site_type_mapping": {
        "0": "non-site",
        "1": "BINDING",
        "2": "ACT_SITE",
        "3": "SITE"
      }
    },
    {
      "id": "protein_2",
      ...
    }
  ],
  "total": 4,
  "successful": 3,
  "failed": 1,
  "failed_ids": ["protein_5"]
}
```

## 输入验证

系统会自动验证输入：

1. ✅ JSON 文件必须是有效的 JSON 格式
2. ✅ 必须是数组格式
3. ✅ 每个条目必须有 `id` 字段
4. ✅ 每个条目必须至少有 `enzyme_structure` 或 `enzyme_sequence` 之一
5. ✅ 如果提供 `enzyme_structure`，文件路径必须存在（在单个预测时）

## 错误处理

- 如果某个蛋白质预测失败，系统会继续处理其他蛋白质
- 失败的蛋白质 ID 会在 `failed_ids` 中列出
- 使用 `--verbose` 标志可以查看详细的错误信息

## 性能提示

1. **内存优化**：使用 `--model-to-use` 只加载需要的模型
   ```bash
   python easifa_predict.py --batch-input batch.json --model-to-use wo_rxn_structures
   ```

2. **序列长度限制**：默认最大序列长度为 1000，可以调整
   ```bash
   python easifa_predict.py --batch-input batch.json --max-length 500
   ```

3. **GPU 加速**：使用 GPU 可以显著提高速度
   ```bash
   python easifa_predict.py --batch-input batch.json --gpu-id 0
   ```

## 示例文件

项目包含一个示例批量输入文件：`batch_input_example.json`

可以使用它测试批量预测功能：

```bash
python easifa_predict.py --batch-input batch_input_example.json --output batch_results.json --verbose
```

## 常见问题

### Q: 批量预测会加载几次模型？

A: 模型只加载一次，然后用于处理所有蛋白质，这样可以节省时间。

### Q: 如果某个蛋白质预测失败会怎样？

A: 系统会记录失败的 ID 并继续处理其他蛋白质，最终输出中会包含失败信息。

### Q: 可以混合使用结构和序列输入吗？

A: 可以！每个蛋白质可以独立选择使用结构或序列作为输入。

### Q: 批量预测的结果如何解析？

A: 结果是一个 JSON 文件，包含 `batch_results` 数组，每个元素对应一个蛋白质的预测结果。可以使用 Python 的 `json` 模块轻松解析：

```python
import json

with open('batch_results.json', 'r') as f:
    results = json.load(f)

for result in results['batch_results']:
    protein_id = result['id']
    predictions = result['predictions']['labels']
    print(f"{protein_id}: {sum(1 for x in predictions if x != 0)} active sites")
```

## 与单个预测对比

| 特性 | 单个预测 | 批量预测 |
|------|---------|---------|
| 输入方式 | 命令行参数 | JSON 文件 |
| 模型加载 | 每次加载 | 加载一次 |
| 错误处理 | 立即失败 | 继续处理 |
| 输出格式 | 单个结果 | 结果数组 |
| 适用场景 | 快速测试 | 大规模预测 |

## 建议

- 对于 1-10 个蛋白质，可以使用单个预测或批量预测
- 对于 10+ 个蛋白质，强烈建议使用批量预测
- 对于大规模预测（100+），考虑分批处理以避免内存问题
