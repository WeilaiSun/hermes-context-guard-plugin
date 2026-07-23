---
name: context-guard--plugin-installer
description: "一键安装 PACE 上下文压缩插件——朋友对 Hermes 说一句话即可完成 clone/patch/install。"
version: "1.0.0"
trigger:
  - "安装上下文压缩"
  - "安装 context guard"
  - "安装 PACE 插件"
  - "安装 context-guard"
---

# context-guard--plugin 一键安装

对着 Hermes 说下面任意一句即可安装：

> 帮我安装 context-guard 上下文压缩插件

安装命令（朋友直接复制发给 Hermes）：

```
从 WeilaiSun/hermes-context-guard-plugin 安装 context-guard 插件到我的 Hermes
```

---

## 安装原理

安装脚本自动完成以下步骤：
1. `git clone` 插件仓库到临时目录
2. 复制 `plugin/` 下所有文件到 `$HERMES_HOME/plugins/context-guard--plugin/`
3. 对 `$HERMES_HOME/hermes-agent/agent/turn_context.py` 打 PACE 源码补丁
4. 在 `config.yaml` 注册插件
5. 安装 Python 依赖（sentence-transformers、torch）
6. 清理临时文件

**安装后需重启 Hermes 桌面端。**

---

## 功能说明

- **PACE 预测式上下文压缩**：每次 LLM 调用前，用 BGE-M3 评分历史片段，动态调整粒度
- **四级粒度**：完整文本 → 详细摘要 → 简短摘要 → 占位符
- **压力自适应**：上下文越满压缩越激进
- **Glimpse 回退**：Agent 可主动请求被压缩片段的完整内容
- **Footnotes**：每次压缩后在消息末尾显示统计

## 前置条件

- Hermes Agent 已安装
- Python 3.9+（Hermes 自带 venv）
- 磁盘空间 ~2GB（BGE-M3 模型首次下载）

## 卸载

删除 `$HERMES_HOME/plugins/context-guard--plugin/` 目录，还原 `turn_context.py` patch（`git checkout agent/turn_context.py`），移除 config.yaml 中的插件注册即可。
