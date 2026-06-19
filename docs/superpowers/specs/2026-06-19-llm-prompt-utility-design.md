# LLM Prompt Utility 设计文档

## 目标

基于 langchain 编写通用大模型调用函数，支持思考模式。覆盖 DeepSeek、GLM、Qwen、MiniMax 四家模型。

## 函数设计

### `call_llm(prompt: str, enable_thinking: bool = True, **kwargs) -> dict`

- `prompt` — 提示词（必填）
- `enable_thinking` — 默认 `True`，启用思考模式
- `**kwargs` — 透传给 ChatOpenAI（如 `temperature`, `model` 覆盖等）

返回值：`{"content": str, "thinking": str | None}`

### 配置加载

- 使用 `python-dotenv` 加载 `.env`（脚本同级目录）
- 读取 `API_KEY`, `BASE_URL`, `MODEL` 三项
- `BASE_URL` 为空时默认用 OpenAI 官方地址

### 思考模式

四家均使用 OpenAI 兼容接口，通过 `extra_body={"thinking": {}}` 传入，从 `AIMessage.response_metadata` 或 `additional_kwargs` 提取 `reasoning_content`。不支持的模型自动降级，`thinking` 返回 `None`。

## 文件结构

```
llm_prompt/
├── llm_prompt.py
└── requirements.txt
```

## main 测试

- 加载 `.env`
- 从 `sys.argv[1]` 取 prompt（默认 "请简要介绍你自己"）
- 支持 `--no-thinking` 关闭思考模式
- 分别打印 thinking 和 content

## 边界处理

- `.env` 缺失或配置不全 → 明确报错
- 不支持思考的模型 → `thinking` 返回 `None`
- 网络/API 异常 → 抛出原始异常
