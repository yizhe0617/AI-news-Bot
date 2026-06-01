# AI Daily Digest

每天抓取 AI 官方部落格與 arXiv 新論文，使用 Gemini API 整理成繁體中文摘要，並送到 Discord Webhook。

## 本機執行

```bash
pip install -r requirements.txt
python main.py --dry-run
```

若要真的送到 Discord：

```bash
set GEMINI_API_KEY=你的_key
set DISCORD_WEBHOOK_URL=你的_webhook_url
python main.py
```

PowerShell 可用：

```powershell
$env:GEMINI_API_KEY="你的_key"
$env:DISCORD_WEBHOOK_URL="你的_webhook_url"
python main.py
```

## Discord Webhook

1. 到 Discord 頻道設定。
2. 選「整合」或「Integrations」。
3. 建立 Webhook。
4. 複製 Webhook URL。
5. 放到 GitHub Secrets 的 `DISCORD_WEBHOOK_URL`。

## GitHub Secrets

在 GitHub repo 裡新增：

- `GEMINI_API_KEY`
- `DISCORD_WEBHOOK_URL`

可選：

- `GEMINI_MODEL`，預設是 `gemini-2.5-flash-lite`。

## 每日排程

`.github/workflows/daily-ai-digest.yml` 會在每天 UTC 00:00 執行，也就是台灣時間早上 8 點。

報告會存到：

```text
reports/YYYY-MM-DD.md
```

已處理過的項目會記在：

```text
data/seen.json
```
