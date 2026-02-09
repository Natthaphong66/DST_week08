# PyCaret MCP Server
## 🛠️ Installation

### Prerequisites
- Python 3.10-3.12
- uv package manager

### Install Dependencies
```bash
cd /path/to/DST_week08
uv sync
```

## 🚀 Usage

### Run MCP Dev Server (Testing)
```bash
cd pycaret_mcp_server
uv run mcp dev server.py
```
เปิด browser ไปที่ http://localhost:5173 เพื่อใช้ MCP Inspector

### Claude Desktop Configuration
เพิ่มใน `~/.config/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "pycaret-server": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "python", "/path/to/pycaret_mcp_server/server.py"],
      "cwd": "/path/to/pycaret_mcp_server"
    }
  }
}
```

## 📚 MCP Tools

### Data Loading
| Tool | Description |
|------|-------------|
| `load_dataset_tool` | โหลด CSV/Excel และแสดง metadata |

### Classification
| Tool | Description |
|------|-------------|
| `setup_classification_tool` | ตั้งค่า classification experiment |
| `compare_classification_models_tool` | เปรียบเทียบ models ทั้งหมด |
| `create_classification_model_tool` | สร้าง model เฉพาะ (lr, rf, xgboost, etc.) |
| `tune_classification_model_tool` | Tune hyperparameters |
| `predict_classification_tool` | ทำนายด้วย model ที่ train แล้ว |
| `save_classification_model_tool` | บันทึก model |

### Regression
| Tool | Description |
|------|-------------|
| `setup_regression_tool` | ตั้งค่า regression experiment |
| `compare_regression_models_tool` | เปรียบเทียบ models ทั้งหมด |
| `create_regression_model_tool` | สร้าง model เฉพาะ |
| `tune_regression_model_tool` | Tune hyperparameters |
| `predict_regression_tool` | ทำนายด้วย model ที่ train แล้ว |
| `save_regression_model_tool` | บันทึก model |

### Utility
| Tool | Description |
|------|-------------|
| `get_available_models_tool` | แสดงรายการ models ที่ใช้ได้ |

## 📁 Project Structure

```
pycaret_mcp_server/
├── server.py              # MCP server entry point
├── core/
│   ├── config.py          # Configuration & FastMCP init
│   ├── data_loader.py     # Data loading utilities
│   ├── classification.py  # Classification functions
│   └── regression.py      # Regression functions
└── logs/                  # Server logs
```

## 📄 License

MIT License
