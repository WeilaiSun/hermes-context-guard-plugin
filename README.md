# context-guard--plugin

PACE (Predictive Adaptive Context Extraction) 上下文压缩插件 for Hermes Agent。

## 一键安装

对着 Hermes 说：

> 从 WeilaiSun/hermes-context-guard-plugin 安装 context-guard 插件

或手动：

```bash
git clone https://github.com/WeilaiSun/hermes-context-guard-plugin.git
cd hermes-context-guard-plugin/plugin
python install.py
```

**安装后需重启 Hermes。**

## 功能

- **PACE 预测式上下文压缩**：每次 LLM 调用前，BGE-M3 评分历史片段，动态选粒度
- **四级粒度**：完整文本 → 详细摘要 → 简短摘要 → 占位符
- **压力自适应**：上下文越满，压缩越激进
- **Footnotes**：每次压缩后在消息末尾显示 `📦 上下文压缩: N 个片段 → X tokens (压力指数 0.67)`

## 论文

Wei et al., "PACE: Predictive Adaptive Context Extraction for Long-Horizon LLM Agents", ACL 2026

## 卸载

```bash
python install.py --uninstall
```

或手动：删除 `$HERMES_HOME/plugins/context-guard--plugin/`，`git checkout agent/turn_context.py`，移除 config.yaml 中插件注册。

## License

MIT
