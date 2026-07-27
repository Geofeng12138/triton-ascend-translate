# 文档翻译工作流说明

## 概述

本文档项目采用 **Sphinx gettext i18n 工作流** 实现中文到英文的自动翻译。核心思想是：源文档为中文（Markdown/RST，位于 `docs/zh/`），通过 Sphinx 的 gettext 构建器提取待翻译字符串（`.pot` 文件），再通过 DeepSeek API 自动翻译生成 `.po` 文件，最后 Read the Docs 在构建英文版时自动应用这些 `.po` 文件渲染出英文 HTML。

## 目录结构与文件功能

### 源文档目录

| 目录/文件 | 功能 |
|-----------|------|
| `docs/zh/` | 中文源文档（Markdown/RST 格式），所有翻译以此为输入 |
| `docs/zh/conf.py` | 中文文档的 Sphinx 配置文件。定义了 `locale_dirs = ['../locale/']` 指向 `docs/locale/`，启用 gettext 国际化支持 |

### 国际化文件目录

| 目录/文件 | 功能 |
|-----------|------|
| `locale/zh/LC_MESSAGES/` | `.pot` 模板文件存放目录。由 `sphinx-build -b gettext` 从 `docs/zh/` 中的中文源文档提取生成 |
| `locale/en/LC_MESSAGES/` | `.po` 翻译文件存放目录。由翻译脚本读取 `.pot` 文件后调用 DeepSeek API 生成 |
| `locale/` | gettext 文件根目录 |

### 翻译脚本文件

| 文件 | 功能 |
|------|------|
| `.github/workflows/scripts/translate_md.py` | 核心翻译脚本。执行完整的翻译流水线：生成 `.pot` → 比对已有翻译 → 调用 DeepSeek API 翻译新内容 → 生成 `.po` 文件 |

### 构建配置文件

| 文件 | 功能 |
|------|------|
| `docs/conf_en.py` | 英文文档构建的 Sphinx 配置文件。源文档目录设置为 `docs/zh/`，`language='en'`，`locale_dirs = ['../locale/']`。Sphinx 会自动找到 `locale/en/LC_MESSAGES/` 下的 `.po` 文件并应用翻译 |
| `.readthedocs.yaml` | Read the Docs 配置文件。中文构建使用 `docs/zh/conf.py`；英文构建通过 `translations` 配置指向 `docs/conf_en.py` |

### CI/CD 配置文件

| 文件 | 功能 |
|------|------|
| `.github/workflows/schedule_doc_translate.yaml` | GitHub Actions 工作流。定时触发翻译任务、提交翻译文件并创建 PR |

## 翻译流程详解

### 整体流程

```
docs/zh/ (中文源文档)
    │
    ▼
sphinx-build -b gettext (提取待翻译字符串)
    │
    ▼
locale/zh/LC_MESSAGES/*.pot (模板文件)
    │
    ▼
translate_md.py (调用 DeepSeek API 翻译)
    │
    ▼
locale/en/LC_MESSAGES/*.po (翻译文件)
    │
    ▼
sphinx-build -D language=en (构建英文文档)
    │
    ▼
locale/en/LC_MESSAGES/*.po (自动被 Sphinx 读取应用)
    │
    ▼
英文 HTML 输出
```

### 步骤 1：生成 .pot 文件

`translate_md.py` 调用 `sphinx-build -b gettext docs/zh locale/` 命令，Sphinx 会扫描 `docs/zh/` 目录下所有的文档，提取需要翻译的文本字符串，生成 `.pot` 文件。

由于 Sphinx 默认将 `.pot` 文件输出到 `locale/` 根目录下（例如 `locale/quick_start.pot`），脚本中的 `_organize_pot_files()` 函数会将这些 `.pot` 文件移动到 `locale/zh/LC_MESSAGES/` 目录下，保持原有的子目录结构。

例如：
- `docs/zh/quick_start.md` → `locale/zh/LC_MESSAGES/quick_start.pot`
- `docs/zh/community/CONTRIBUTING_zh.md` → `locale/zh/LC_MESSAGES/community/CONTRIBUTING_zh.pot`

### 步骤 2：翻译新内容

`translate_md.py` 逐一处理每个 `.pot` 文件：

1. 解析 `.pot` 文件获取所有待翻译条目（msgid）
2. 对比已存在的 `.po` 文件，复用已有的翻译（翻译记忆功能）
3. 对新增或未翻译的条目，调用 DeepSeek API 进行翻译
4. 将翻译结果写入 `.po` 文件，存放到 `locale/en/LC_MESSAGES/` 目录，保持与源文档相同的子目录结构

例如：
- `locale/zh/LC_MESSAGES/quick_start.pot` → `locale/en/LC_MESSAGES/quick_start.po`
- `locale/zh/LC_MESSAGES/community/CONTRIBUTING_zh.pot` → `locale/en/LC_MESSAGES/community/CONTRIBUTING_zh.po`

### 步骤 3：构建英文文档

Read the Docs 在构建英文版本时，使用 `docs/conf_en.py` 作为配置文件：
- 源目录指向 `docs/zh/`（中文源文档）
- `language = 'en'`（目标语言为英文）
- `locale_dirs = ['../locale/']`（指向 `docs/locale/`，即项目根目录的 `locale/` 文件夹）

Sphinx 会自动读取 `locale/en/LC_MESSAGES/` 下的 `.po` 文件，将中文源文档中的文本替换为对应的英文翻译，生成英文 HTML 页面。

### 用户访问流程

当用户通过浏览器访问 `https://triton-ascend.readthedocs.io/en` 时：
1. Read the Docs 检测到语言为英文
2. 使用 `docs/conf_en.py` 配置进行构建
3. 源文档从 `docs/zh/` 读取（中文源）
4. Sphinx 自动从 `locale/en/LC_MESSAGES/` 加载对应的 `.po` 翻译文件
5. 中文文本被替换为英文翻译，最终渲染出英文文档页面

## 不需要翻译的目录和文件

以下目录和文件会被排除，不进行翻译：

| 排除项 | 类型 | 原因 |
|--------|------|------|
| `python-api/` | 目录 | Python API 文档，保持原样 |
| `triton_api/` | 目录 | Triton API 文档，保持原样 |
| `triton_api_extension/` | 目录 | Triton API 扩展文档，保持原样 |
| `libdevice/` | 目录 | Libdevice 文档，保持原样 |
| `code_of_conduct.md` | 文件 | 行为准则，保持原样 |

这些排除配置在 `translate_md.py` 的 `EXCLUDED_DIRS` 和 `EXCLUDED_FILES` 常量中定义。

## 触发时机

### 自动触发（定时）

GitHub Actions 工作流会根据 cron 表达式 `0 0 */3 * *` **每 3 天**自动执行一次翻译任务。即每月大约执行 10 次。

### 手动触发

也可通过 GitHub Actions 界面手动触发工作流，支持以下参数：

| 参数 | 说明 |
|------|------|
| `target_branch` | PR 的目标分支，默认为 `main` |
| `full_translate` | 是否进行全量翻译。若为 `true` 则翻译所有文档；若为 `false`（默认）则仅增量翻译新增/变更部分 |

### 增量 vs 全量翻译

| 模式 | 条件 | 行为 |
|------|------|------|
| 全量翻译 | `full_translate=true` 或首次运行（`locale/en/LC_MESSAGES/` 为空） | 翻译所有非排除的 `.pot` 文件 |
| 增量翻译 | 默认模式 | 通过 git diff 检测 `locale/zh/LC_MESSAGES/` 中新增或变更的 `.pot` 文件，仅翻译差异部分 |

## 翻译脚本使用方式

```bash
# 全量翻译（生成 .pot 并翻译所有文件）
python .github/workflows/scripts/translate_md.py --first-time

# 增量翻译（仅翻译新增/变更的 .pot 文件）
python .github/workflows/scripts/translate_md.py --all

# 跳过 gettext 生成步骤（直接翻译已有的 .pot 文件）
python .github/workflows/scripts/translate_md.py --all --skip-gettext
```

环境变量要求：
- `DEEPSEEK_API_KEY`：DeepSeek API 密钥，用于调用翻译接口

## PR 提交规范

工作流自动创建的 PR 遵循以下规范：

| 规范项 | 格式 |
|--------|------|
| 分支名 | `auto-pr/doc-translate-YYYYMMDDHHMMSS` |
| 提交消息 | `[doc](feat) Auto-translate N file(s) (Sphinx gettext)` |
| PR 标题 | `[doc](feat) Auto-translate Chinese docs to English YYYY-MM-DD`（长度 ≤ 72 字符） |
| PR 标题必须匹配 | `^\[[^]]+\]\((feat\|fix\|NFC\|style\|test\|chore\|revert)\) .+$` |

## 文件示例

以下是一个典型的 `.po` 文件内容示例（`locale/en/LC_MESSAGES/quick_start.po`）：

```po
# English translations for triton-ascend docs.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
#
msgid ""
msgstr ""
"Project-Id-Version: triton-ascend-docs\n"
"POT-Creation-Date: 2026-07-27 14:30+0800\n"
"PO-Revision-Date: 2026-07-27 14:30+0800\n"
"Last-Translator: Auto Translation (DeepSeek)\n"
"Language-Team: English\n"
"Language: en\n"
"MIME-Version: 1.0\n"
"Content-Type: text/plain; charset=UTF-8\n"
"Content-Transfer-Encoding: 8bit\n"
"Plural-Forms: nplurals=2; plural=(n != 1);\n"

msgid "快速开始"
msgstr "Quick Start"

msgid "环境准备"
msgstr "Environment Setup"