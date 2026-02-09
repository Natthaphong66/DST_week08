# PyCaret MCP Server
## 🚀 การติดตั้ง

```bash
# ติดตั้ง dependencies
uv sync

# รัน server
uv run python -m pycaret_mcp_server.server
```

## 🛠️ MCP Tools ที่มี

### Data Loading
| Tool | หน้าที่ |
|------|--------|
| `load_dataset_tool` | โหลดข้อมูล CSV/Excel |

### Classification
| Tool | หน้าที่ |
|------|--------|
| `setup_classification_tool` | ตั้งค่า classification experiment |
| `compare_classification_models_tool` | เปรียบเทียบ models ทั้งหมด |
| `create_classification_model_tool` | สร้าง model ที่เลือก |
| `tune_classification_model_tool` | ปรับ hyperparameters |
| `predict_classification_tool` | ทำนายข้อมูลใหม่ |
| `save_classification_model_tool` | บันทึก model |

### Regression
| Tool | หน้าที่ |
|------|--------|
| `setup_regression_tool` | ตั้งค่า regression experiment |
| `compare_regression_models_tool` | เปรียบเทียบ models ทั้งหมด |
| `create_regression_model_tool` | สร้าง model ที่เลือก |
| `tune_regression_model_tool` | ปรับ hyperparameters |
| `predict_regression_tool` | ทำนายข้อมูลใหม่ |
| `save_regression_model_tool` | บันทึก model |

### Utility
| Tool | หน้าที่ |
|------|--------|
| `get_available_models_tool` | แสดงรายชื่อ models ที่มี |

## 🔧 การใช้งาน

### 1. ทดสอบด้วย MCP Inspector
```bash
uv run mcp dev pycaret_mcp_server/server.py
```
เปิด browser ที่ `http://localhost:5173`

### 2. ใช้กับ Claude Desktop
เพิ่ม config ในไฟล์ `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pycaret": {
      "command": "uv",
      "args": ["run", "python", "-m", "pycaret_mcp_server.server"],
      "cwd": "/path/to/DST_week08"
    }
  }
}
```

## 📁 โครงสร้างโปรเจกต์

```
pycaret_mcp_server/
├── __init__.py
├── server.py              # Main MCP server
├── README.md
└── core/
    ├── __init__.py
    ├── config.py          # Configuration & FastMCP init
    ├── data_loader.py     # Data loading utilities
    ├── classification.py  # Classification functions
    └── regression.py      # Regression functions
```

## 📦 Dependencies

- Python >=3.10, <3.13
- PyCaret
- FastMCP
- chardet
- psutil
