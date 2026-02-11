# PyTorch Data Loaders

纯PyTorch张量的时间序列数据加载器,从Time-Series-Library适配而来。

## 特性

- ✅ **纯PyTorch张量**: 所有数据加载器直接返回`torch.Tensor`,无需numpy转换
- ✅ **标准归一化**: 使用`StandardScaler`进行数据归一化
- ✅ **多种分割**: 支持train/val/test自动分割
- ✅ **时间编码**: 两种模式 - categorical (月/日/时) 和 time_features
- ✅ **多种特征模式**: M (多变量), S (单变量), MS (多变量到单变量)
- ✅ **工厂模式**: 通过名称动态创建数据加载器
- ✅ **自定义数据集**: 支持任意CSV格式

## 数据集类

### ETTHourDataset

Electricity Transformer Temperature (ETT) 小时级数据集。

```python
from liulian.data.torch_datasets import ETTHourDataset

dataset = ETTHourDataset(
    root_path='./data/ETT',
    data_path='ETTh1.csv',
    flag='train',  # 'train', 'val', 'test'
    size=(96, 48, 96),  # (seq_len, label_len, pred_len)
    features='M',  # 'M', 'S', 'MS'
    scale=True,
)

# Get a sample - all torch tensors!
seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[0]
```

**分割配置**:
- Train: 前12个月
- Val: 接下来4个月  
- Test: 最后4个月

### ETTMinuteDataset

ETT 15分钟级数据集 (与ETTHourDataset类似,但支持分钟级时间特征)。

```python
from liulian.data.torch_datasets import ETTMinuteDataset

dataset = ETTMinuteDataset(
    root_path='./data/ETT',
    data_path='ETTm1.csv',
    flag='train',
    freq='t',  # 15-minute frequency
)
```

### CustomCSVDataset

通用CSV数据集,支持任意CSV格式。

```python
from liulian.data.torch_datasets import CustomCSVDataset

dataset = CustomCSVDataset(
    root_path='./data',
    data_path='my_data.csv',
    flag='train',
    target='target_column',  # Your target column name
    train_ratio=0.7,  # Customizable split
    test_ratio=0.2,
)
```

**CSV要求**:
- 必须包含 `date` 列
- 必须包含目标列 (通过`target`参数指定)
- 其他列自动作为特征

## 工厂函数

### create_dataloader

创建单个DataLoader:

```python
from liulian.data.data_factory import create_dataloader

loader = create_dataloader(
    data_name='ETTh1',  # 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'custom'
    root_path='./data/ETT',
    data_path='ETTh1.csv',
    flag='train',
    size=(96, 48, 96),
    batch_size=32,
    shuffle=True,
)

# Iterate through batches
for batch in loader:
    seq_x, seq_y, seq_x_mark, seq_y_mark = batch
    # All are torch.Tensor!
```

### create_dataloaders

一次创建所有分割:

```python
from liulian.data.data_factory import create_dataloaders

loaders = create_dataloaders(
    data_name='ETTh1',
    root_path='./data/ETT',
    data_path='ETTh1.csv',
    size=(96, 48, 96),
    batch_size=32,
)

train_loader = loaders['train']
val_loader = loaders['val']
test_loader = loaders['test']
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `root_path` | str | - | 数据文件根目录 |
| `data_path` | str | - | CSV文件名 |
| `flag` | str | 'train' | 分割类型: 'train', 'val', 'test' |
| `size` | tuple | (96, 48, 96) | (seq_len, label_len, pred_len) |
| `features` | str | 'M' | 特征模式: 'M', 'S', 'MS' |
| `target` | str | 'OT' | 目标列名称 |
| `scale` | bool | True | 是否使用StandardScaler归一化 |
| `timeenc` | int | 0 | 时间编码: 0 (categorical), 1 (time_features) |
| `freq` | str | 'h' | 频率字符串: 'h' (小时), 't' (15分钟) |
| `batch_size` | int | 32 | DataLoader批大小 |
| `shuffle` | bool | True | 是否打乱数据 |

## 返回格式

所有数据加载器返回4个torch.Tensor:

```python
seq_x, seq_y, seq_x_mark, seq_y_mark = batch

# seq_x: 输入序列 [batch_size, seq_len, features]
# seq_y: 目标序列 [batch_size, label_len + pred_len, features]  
# seq_x_mark: 输入时间特征 [batch_size, seq_len, time_dim]
# seq_y_mark: 目标时间特征 [batch_size, label_len + pred_len, time_dim]
```

## 与模型集成

数据加载器返回的张量可以直接传递给PyTorch模型适配器:

```python
from liulian.data.data_factory import create_dataloader
from liulian.models.torch.dlinear_adapter import DLinearAdapter

# Load data
loader = create_dataloader(
    data_name='ETTh1',
    root_path='./data/ETT',
    data_path='ETTh1.csv',
    flag='test',
    batch_size=32,
)

# Create model adapter
model_adapter = DLinearAdapter(model, config)

# Run inference - NO numpy conversion!
for batch in loader:
    seq_x, seq_y, seq_x_mark, seq_y_mark = batch
    
    # Prepare model input
    model_input = {
        'x_enc': seq_x,
        'x_mark_enc': seq_x_mark,
        'x_dec': seq_y[:, :config['label_len'], :],
        'x_mark_dec': seq_y_mark,
    }
    
    # Forward pass - pure torch tensors throughout!
    output = model_adapter.forward(model_input)
    predictions = output['predictions']
```

## 测试

运行完整测试套件:

```bash
pytest tests/data/test_torch_datasets.py -v
```

**测试覆盖**:
- ✅ 基本加载功能
- ✅ 返回torch.Tensor类型
- ✅ 张量形状验证
- ✅ 单变量/多变量模式
- ✅ inverse_transform
- ✅ train/val/test分割
- ✅ 工厂模式
- ✅ 端到端pipeline

**测试结果**: 18/18 通过 ✅

## 文件结构

```
liulian/data/
├── torch_datasets.py       # Dataset classes (~560 lines)
│   ├── ETTHourDataset
│   ├── ETTMinuteDataset
│   └── CustomCSVDataset
└── data_factory.py         # Factory functions (~220 lines)
    ├── create_dataloader()
    ├── create_dataloaders()
    └── register_dataset()

tests/data/
└── test_torch_datasets.py  # Test suite (~550 lines, 18 tests)

examples/
└── data_loaders_example.py # Usage examples
```

## 注册自定义数据集

可以注册自己的数据集类:

```python
from liulian.data.data_factory import register_dataset
from torch.utils.data import Dataset

class MyCustomDataset(Dataset):
    def __init__(self, root_path, data_path, flag, **kwargs):
        # Your implementation
        pass
    
    def __getitem__(self, index):
        # Return (seq_x, seq_y, seq_x_mark, seq_y_mark)
        pass
    
    def __len__(self):
        return self.num_samples

# Register
register_dataset('my_dataset', MyCustomDataset)

# Use
loader = create_dataloader('my_dataset', ...)
```

## 依赖项

- Python 3.12+
- PyTorch 2.10+
- pandas
- numpy
- scikit-learn

## 许可

MIT License (适配自Time-Series-Library)

## 架构决策

**为什么不转换numpy?**  
传统实现在数据加载后转换为numpy,然后在模型输入时再转换回torch。这会:
- 增加不必要的开销
- 破坏梯度流
- 使代码更复杂

我们的实现直接返回torch.Tensor,整个pipeline保持纯PyTorch:
```
CSV → pandas → torch.Tensor → Model → torch.Tensor
```

**COPY-PASTE原则**:  
我们从参考项目复制了核心逻辑,但进行了torch适配,而不是创建抽象层。这确保了:
- 与参考实现的一致性
- 更容易调试和维护
- 清晰的代码溯源
