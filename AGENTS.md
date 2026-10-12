# 仓库协作规则（给所有 AI 助手：Claude Code、Codex 等）

## 跨平台：Windows 与 Linux 都必须能跑

用户会在 Windows 和 Linux（Jetson / Ubuntu）之间来回切换，并用 git 同步。本仓库里写或改的代码、脚本、配置，必须两个平台都能运行，git 也要能正常同步。

- 路径用 `pathlib` 或相对路径；不写死 `E:\`，不写反斜杠。
- 换行符统一 LF，遵守仓库 `.gitattributes`；不要引入 CRLF。
- 文件名区分大小写（Linux 敏感），引用时大小写要与实际一致。
- Windows 专属的东西（`msvcrt`、`ctypes.windll`、`.dll`、`os.add_dll_directory`、`timeBeginPeriod`、PowerShell/批处理）必须用 `sys.platform` 判断，并提供 Linux 分支（`.so`、`.sh`）。
- 不提交本机路径、`__pycache__`、日志、缓存；大文件走 Git LFS（见 `.gitattributes`）。
- 改动后在能跑的平台上自测；不能在另一平台验证的部分，在说明里写清楚。

## 安全

- 未经用户当次明确同意，不使能电机、不发送运动命令。
- 不改训练中正在使用的模型/配置；先给出修改建议。
