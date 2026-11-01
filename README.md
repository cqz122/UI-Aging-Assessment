# 适老化评估

给一张 App 截图，按检测条目逐条判定，输出一份反馈文档。

一条检测条目一个文件，各写各的，互不覆盖；新增测量工具同样是新建一个文件，程序会自动发现。

---

## 一、项目结构

```text
适老化评估/
├── configs/
│   └── config.py           # 全局配置：截图 dpi、对比度阈值、模型名
├── core/                   # 调度核心，一般不用改
│   ├── agent.py            # 逐条判定：跑工具 / 问模型 / 汇总
│   ├── context.py          # 读图、跑 OCR、生成 page 对象
│   ├── llm.py              # 调用大模型（OpenAI 兼容），取回 JSON
│   └── report.py           # 生成反馈文档、导出条目清单
├── prompts/                # ★ 检测条目，一个文件一条
│   ├── _模板.py            # 新建条目时复制这个（以 _ 开头不会被加载）
│   ├── A03_对比度.py       # 示例：不用大模型，直接比数值
│   └── A05_广告弹窗.py     # 示例：让大模型判断
├── tools/                  # ★ 测量工具，一个文件一个
│   ├── contrast.py         # 参考实现：对比度
│   ├── ocr.py              # PaddleOCR 封装，不用改
│   └── __init__.py         # 自动扫描工具，不用改
├── data/
│   └── screenshots/        # 截图放这里
├── eval/
│   ├── recall.py           # 召回率：系统结果 vs 人工标注
│   └── labels/             # 人工标注，一人一个 json
├── examples/               # 演示截图与示例反馈文档
├── outputs/                # 运行结果（已 gitignore）
├── main.py                 # 运行入口
├── requirements.txt
├── .env.example            # 复制成 .env 填 API Key
└── .gitignore
```

程序启动时会扫描 `prompts/` 和 `tools/`，新建的文件自动加载，**不需要登记**。

---

## 二、安装与运行

### 2.1 装依赖

需要 **Python 3.10 或 3.11**（PaddlePaddle 对 3.12+ 支持不完整）。

```bash
git clone <仓库地址>
cd 适老化评估

python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt`：

```text
paddlepaddle==2.6.2        # OCR 推理框架
paddleocr==2.9.1           # 文字识别
numpy<2
opencv-python-headless     # 图像处理
openai>=1.30               # 调用大模型
```

### 2.2 配置 Key

```bash
copy .env.example .env      # macOS/Linux: cp .env.example .env
```

打开 `.env` 填入：

```ini
VLM_API_KEY=你的Key
VLM_BASE_URL=https://api.deepseek.com
VLM_MODEL=deepseek-flash
```

换别的厂商只改这三行，代码不用动，例如火山方舟（豆包）：

```ini
VLM_API_KEY=你的火山方舟Key
VLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
VLM_MODEL=ep-20250101-xxxxx
```

`VLM_MODEL` 要填方舟控制台里"接入点"的 ID（`ep-` 开头），不是模型名。

> `.env` 已加入 `.gitignore`，不会被提交。不要把 Key 写进代码。

### 2.3 跑一张图

截图统一用 **mumu 模拟器，机型 vivo X90s（V2241A），1080 × 1920，DPI 480**，
放在 `data/screenshots/` 下，命名成 `01_首页.png`。

```bash
python main.py data/screenshots/01_首页.png
```

首次运行会下载 OCR 模型（约 20MB），之后离线可用。结果在 `outputs/`。

常用命令：

```bash
python main.py --list-items                # 有哪些条目
python main.py --list-tools                # 有哪些工具
python main.py 图.png --items A05          # 只跑指定条目
python main.py 图.png --print-facts        # 打印模型看到的页面事实卡
python main.py 图.png --mock               # 不调模型，只跑程序测量
python main.py --export-items              # 汇总所有条目成一份清单
```

---

## 三、写一条检测条目

一条条目 = `prompts/` 下的一个 `.py` 文件，文件里只有一个字典 `ITEM`。

- 模板：[`prompts/_模板.py`](prompts/_模板.py)
- 现成例子：[`prompts/A05_广告弹窗.py`](prompts/A05_广告弹窗.py)（让大模型判断）
- 现成例子：[`prompts/A03_对比度.py`](prompts/A03_对比度.py)（不用大模型，直接比数值）

新建 `prompts/A06_你的短名.py`，把模板内容复制进去改就行。文件名以 `_` 开头的不会被加载。

### 3.1 一条条目包含哪些字段

"字段"就是字典里的键。一条条目由下面这些字段组成，带 ★ 的必须写：

| 字段名 | 是什么 | 必填 | 例子 |
| --- | --- | --- | --- |
| `id` | 条目编号，不能和别人重复 | ★ | `"A06"` |
| `title` | 条目名称，一句话 | ★ | `"不得出现诱导下载、诱导付款等诱导式按键"` |
| `condition` | 适用条件：什么页面、什么状态下才成立 | ★ | `"适用于所有页面，尤其支付、下载入口附近"` |
| `source` | 依据来源：规范名 + 条款号 | ★ | `"工信部……规范 5.1.2"` |
| `method` | 判定方式，三选一：`"llm"` 让模型判断 / `"tool"` 纯数值比阈值 / `"hybrid"` 先测再让模型判 | ★ | `"llm"` |
| `prompt` | 判定要求正文；`method` 含 `"llm"` 时必填 | ★（llm） | 见 3.2 |
| `examples` | 判定示例（few-shot），1~3 个"输入 → 输出"对 | 强烈建议 | 见 3.3 |
| `tool` | 用哪个工具测；`method` 含 `"tool"` 时必填 | ★（tool） | `"contrast"` |
| `judge` | 拿工具的哪个字段、跟什么阈值比；`method` 为 `"tool"` 时必填 | ★（tool） | 见 3.4 |
| `suggestion` | 判不通过时，报告里给出的默认建议 | 否 | `"移除该横幅"` |

`method` 决定这条怎么跑：

```text
method = "llm"     条目要求 + 判定示例 + 截图 + 页面事实卡 → 大模型 → JSON
method = "tool"    工具算出的数值 → 和 judge 里的阈值比 → 直接出结论
method = "hybrid"  先跑工具，再把数值一起交给大模型判断
```

### 3.2 判定要求（prompt）怎么写

四段就够，按顺序写：

```text
1. 判"不通过"的情形（逐条列出，越具体越稳定）
2. 判"通过"的情形（给出反例，防止模型过度敏感）
3. 判"无法判断"的情形（看不到的信息不要硬判）
4. 证据要求（引用截图位置或事实卡原文）
```

写"什么算不通过"，**不要**写"请评估这一条是否友好"——后者只能得到客套话。

参考 [`prompts/A05_广告弹窗.py`](prompts/A05_广告弹窗.py) 里的 `prompt` 字段。

### 3.3 判定示例（examples）怎么写

模型会照着示例的**格式、措辞和严格程度**回答。没有示例时，同一条目跑两次可能给出不同风格的结论。

`examples` 是一个列表，每个元素有 `input`（描述看到了什么）和 `output`（标准答案）：

```json
{
  "input": "截图：页面底部有「立即领取 88 元红包」的横幅，右上角有「跳过」按钮。",
  "output": {
    "item_id": "A06",
    "status": "不通过",
    "evidence": ["页面底部出现「立即领取 88 元红包」"],
    "reason": "出现诱导性文案与关闭按钮",
    "suggestion": "移除该横幅"
  }
}
```

`output` 的字段固定为四个（`status` 只能是"通过 / 不通过 / 无法判断"）：

| 字段 | 含义 |
| --- | --- |
| `item_id` | 条目编号，和 `id` 一致 |
| `status` | 结论：通过 / 不通过 / 无法判断 |
| `evidence` | 证据列表，引用截图位置或事实卡原文 |
| `reason` | 一句话说明理由 |
| `suggestion` | 怎么改，判通过时可以留空 |

至少给两个示例：一个"不通过"、一个"通过"，模型才知道边界在哪。

`method: "hybrid"` 的条目可以把工具算出的数值放进 `input`，这是最有用的示例形式：

```text
input：工具返回 {"body_dp": 12.95, "threshold_dp": 18.0}
output：{"item_id": "A01", "status": "不通过",
         "reason": "工具测得正文字号 12.95dp，低于 18dp 阈值", ...}
```

### 3.4 不用大模型的条目：直接比数值

这类条目没有 `prompt`，靠 `judge` 声明"拿哪个字段、跟什么比"。完整例子见
[`prompts/A03_对比度.py`](prompts/A03_对比度.py)：

```python
"method": "tool",
"tool": "contrast",                     # 工具名
"judge": {
    "field": "low_count",               # 取工具返回字典里的哪个键
    "op": "==",                         # 比较方式：>= > <= < ==
    "value": 0,                         # 阈值
    "unit": " 行",                       # 单位，只用于报告显示
    "label": "低于 4.5:1 的文字行数",     # 中文名，只用于报告显示
},
```

`field` 写什么，取决于工具返回什么——见第四节 4.4。

### 3.5 写完自己跑一遍

```bash
python main.py data/screenshots/01_首页.png --items A06 --print-facts
```

- `--items A06` 只跑你这条，快；
- `--print-facts` 打印模型看到的页面事实卡，判不准时先看这里；
- `--mock` 不调用模型（省额度），只看流程通不通。

---

## 四、写一个测量工具

工具 = `tools/` 下的一个 `.py` 文件。参考实现：[`tools/contrast.py`](tools/contrast.py)。

### 4.1 一个工具必须写哪三样

```python
NAME = "contrast"                                   # 工具名，条目里用它引用
DESCRIPTION = "测量文字与背景的对比度（WCAG 2.1）"    # 一句话说明
def run(page) -> dict:                              # 输入 page，输出字典
    ...
```

写了就生效：程序启动时扫描 `tools/`，看到带 `NAME` 和 `run()` 的文件就自动加载，
**不用改 `tools/__init__.py`**。文件名以 `_` 开头的会被跳过（可以放草稿）。

### 4.2 `page` 是从哪里来的

`page` 不是你创建的，是程序建好后传给你的：

```text
python main.py data/screenshots/01_首页.png
      │
      ▼
core/agent.py     读入这张图
      │
      ▼
core/context.py   PageContext.load(图片路径)
      │           ├─ 读图
      │           └─ 跑 PaddleOCR，拿到所有文字行
      ▼
tools/contrast.py  run(page)  ← page 从这里进来
```

所以写工具时，你只需要写 `def run(page):`，直接用 `page` 里现成的东西，不用自己读图、不用自己调 OCR。

### 4.3 `page` 里有哪些东西

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `page.image` | `numpy.ndarray` | 截图本体，BGR 顺序，`shape = (高, 宽, 3)` |
| `page.lines` | `List[TextLine]` | OCR 结果，一行文字一个对象 |
| `page.dpi` | `float` | 设备像素密度，换算公式 `dp = px * 160 / dpi` |
| `page.config` | 模块 | 阈值常量，如 `page.config.CONTRAST_MIN` |
| `page.tool("contrast")` | `dict` | 复用别的工具的结果，自动缓存，不会重复算 |

`page.lines` 里每个 `TextLine` 有：

| 属性 | 含义 |
| --- | --- |
| `.text` | 这一行的文字内容 |
| `.x1` `.y1` `.x2` `.y2` | 外接框的四个坐标（像素） |
| `.width` `.height` | 外接框的宽和高 |
| `.box` | 四个坐标组成的元组 `(x1, y1, x2, y2)` |
| `.confidence` | OCR 置信度，0~1 |

### 4.4 工具要返回什么

返回一个**字典**，键名用英文，值用数字或字符串。这个字典会去三个地方：

```text
                     ┌─→ 条目的 judge 取值，程序据此判通过/不通过
工具返回的字典 ───────┼─→ 写进页面事实卡，模型能看到这些数值
                     └─→ 存进结果 JSON，报告与召回率脚本用
```

对比度工具的返回：

```python
{"mean_ratio": 8.2, "min_ratio": 2.05, "line_count": 12,
 "low_count": 4, "threshold": 4.5}
```

字段名要起得能看懂：`mean_ratio` 比 `r1` 好，因为模型也会读到这些名字。

### 4.5 让条目用上你的工具

工具写好后，在条目里引用（`field` 就是 4.4 里你起的键名）：

```python
"method": "tool",
"tool": "你的工具名",
"judge": {"field": "mean_ratio", "op": ">=", "value": 4.5,
          "unit": ":1", "label": "平均对比度"},
```

先 `python main.py --list-tools` 确认工具被识别，再跑一张图看报告里有没有这一条。

---

## 五、结果与评估

### 5.1 输出文件

- `outputs/xxx_反馈.md` —— 反馈文档
- `outputs/xxx_result.json` —— 结构化结果

### 5.2 汇总条目

```bash
python main.py --export-items      # 生成 outputs/条目清单.md
```

### 5.3 人工标注

标注是人工判断的"标准答案"，用来算召回率：系统判"不通过"、你标"通过"，就是误报。

**位置**：`eval/labels/`，**一人一个 json**，文件名用你的名字：

```text
eval/labels/
├── demo.json
├── 张三.json
└── 李四.json
```

**格式**：`{图片文件名: {条目编号: 结论}}`

```json
{
  "01_首页.png": {
    "A03": "不通过",
    "A05": "通过",
    "A06": "无法判断"
  },
  "02_健康档案.png": {
    "A03": "通过"
  }
}
```

四条规矩：

1. **先标注，再看系统结果**——看了结果再标，召回率就没有意义；
2. 只能填 `通过 / 不通过 / 无法判断`，和系统口径一致；
3. "无法判断"也算一种标注，静态截图确实看不出来的就标它；
4. 图片名必须和 `data/screenshots/` 里的**完全一致**，包括扩展名。

### 5.4 召回率与覆盖率

```bash
python eval/recall.py
```

输出命中、漏报、误报，以及召回率、精确率、判定覆盖率。

```
判定覆盖率 = 系统给出明确结论的条目数 / 总条目数
```

"无法判断"不算结论。覆盖率低说明条目设计得太依赖截图里没有的信息，需要改 prompt 或改判定方式。

---

## 六、常见问题

**Q：跑出来全是"无法判断"？**
没有配 `VLM_API_KEY`，或者用了 `--mock`。`--mock` 只跑程序测量的条目。

**Q：提示词改了没效果？**
先 `--print-facts` 看模型到底看到了什么。多数"判不准"是事实卡里没这个信息，而不是提示词写得不好。

**Q：只跑自己那两条？**
`python main.py 截图.png --items A06,A07`。

**Q：`PermissionError: ~/.paddleocr`？**
服务器不让写用户目录。在 `.env` 里加 `AGEFRIENDLY_MODEL_DIR=models`，模型会下到项目内。

**Q：换了设备，dpi 要改吗？**
要。改 `configs/config.py` 的 `DEVICE_DPI`，或在 `.env` 里设 `AGEFRIENDLY_DPI`。
同一批评估必须同一套模拟器参数。

---

## 七、提交前自查

- [ ] 文件在 `prompts/` 下，名字是 `编号_短名.py`，没有改别人的文件
- [ ] `id` 和《检测条目提名表》一致，没有重复
- [ ] `condition` 写清了适用页面和状态
- [ ] `prompt` 里分别写了"不通过 / 通过 / 无法判断"
- [ ] `examples` 至少有一个不通过、一个通过的示例
- [ ] 本地跑过 `python main.py 截图.png --items 你的编号`，结果符合预期
- [ ] 标注写进了 `eval/labels/你的名字.json`

---

## 许可

MIT License
