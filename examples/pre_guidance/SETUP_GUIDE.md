# 🔧 环境配置指南

## 📋 必需的API密钥

本项目需要两个API服务的密钥：

### 1. HHEM (Vectara) API密钥
- **用途**: 事实一致性评估
- **获取地址**: https://vectara.com/
- **设置方式**: 
  - 环境变量: `VECTARA_API_KEY`
  - 或直接在代码中传入

### 2. DashScope (阿里云) API密钥  
- **用途**: 通义千问幻觉检测
- **获取地址**: https://dashscope.console.aliyun.com/
- **设置方式**: 环境变量 `DASHSCOPE_API_KEY`

## ⚡ 快速设置步骤

### 步骤1: 获取DashScope API密钥

1. 访问 [阿里云DashScope控制台](https://dashscope.console.aliyun.com/)
2. 登录你的阿里云账号（没有账号需要先注册）
3. 开通灵积模型服务（DashScope）
4. 在控制台中创建API密钥
5. 复制生成的API密钥

### 步骤2: 设置环境变量

#### macOS/Linux 用户:
```bash
# 临时设置（当前终端会话有效）
export DASHSCOPE_API_KEY="sk-your-api-key-here"

# 永久设置（推荐）
echo 'export DASHSCOPE_API_KEY="sk-your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

#### Windows 用户:
```cmd
# 临时设置
set DASHSCOPE_API_KEY=sk-your-api-key-here

# 永久设置（系统环境变量）
setx DASHSCOPE_API_KEY "sk-your-api-key-here"
```

### 步骤3: 验证设置

运行以下命令验证设置是否正确：
```bash
cd examples/pre_guidance
python3 api_setup_guide.py
```

## 📦 依赖安装

```bash
# 基础依赖
pip install requests

# 阿里云DashScope SDK
pip install dashscope
```

## 🔍 故障排除

### 常见问题

1. **ImportError: No module named 'dashscope'**
   ```bash
   pip install dashscope
   ```

2. **API密钥未设置错误**
   - 检查环境变量是否正确设置
   - 确认密钥格式正确（通常以sk-开头）

3. **网络连接错误**
   - 确认网络连接正常
   - 检查防火墙设置

4. **API调用失败**
   - 验证API密钥有效性
   - 检查API配额是否用完

### 测试连接
```python
# 测试HHEM连接
from src.HHEM_API import HHEMFactualConsistencyAPI
api = HHEMFactualConsistencyAPI()

# 测试DashScope连接  
from src.qwen_hallucination_evaluator import QwenHallucinationEvaluator
evaluator = QwenHallucinationEvaluator()
```

## 🚀 开始使用

配置完成后，你可以：

1. **快速体验**: `python3 examples/quick_start.py`
2. **查看示例**: `python3 examples/practical_examples.py`
3. **运行测试**: `python3 tests/test_integrated_evaluator.py`

## 💡 使用建议

- 🔒 **安全**: 不要在代码中硬编码API密钥
- 💰 **成本**: 注意API调用费用，合理使用
- 🎯 **准确性**: 重要决策建议人工复核评估结果
- 🔄 **备份**: 建议同时使用两种评估方法以提高可靠性
