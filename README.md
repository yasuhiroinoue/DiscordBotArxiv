# ArXiv Research Discord Bot

ArXiv論文を検索して日本語で要約・分析するDiscord Botです。Gemini 2.0を活用して学術研究をサポートします。

## 主な機能

- **学術論文の検索**: ArXivから最新の論文を検索
- **論文の要約と分析**: 論文の内容を日本語で要約・分析
- **統合検索機能**: Google検索結果とArXiv論文を組み合わせた包括的なリサーチ
- **複数の研究モード**: 基本/Web/学術の3種類の研究モードに対応
- **ファイル出力**: 長い研究結果をファイルとして保存可能

## インストール方法

### 必要な環境

- Python 3.8以上
- Discord Bot Token
- Google API Key (Gemini API)
- Google Custom Search Engine ID (オプション、Web検索用)

### セットアップ手順

1. リポジトリをクローン
   ```bash
   git clone https://github.com/yasuhiroinoue/DiscordBotArxiv.git
   cd DiscordBotArxiv
   ```

2. 必要なパッケージをインストール
   ```bash
   pip install -r requirements.txt
   ```

3. `.env`ファイルを作成して環境変数を設定
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   GOOGLE_CSE_ID=your_custom_search_engine_id
   DISCORD_BOT_TOKEN=your_discord_bot_token
   ```

4. Botを起動
   ```bash
   python disco_arxiv.py
   ```

## 必要なライブラリ

```
discord.py
python-dotenv
arxiv
langchain
langchain-google-genai
langchain-community
beautifulsoup4
requests
```

## 使い方

### Discord上でのコマンド

以下のコマンドはすべて`@BotName`でメンションするか、DMで直接実行できます。

- `@Bot basic: <テーマ>` - LLMの知識のみを使用した基本リサーチ
- `@Bot web: <テーマ>` - Web検索を含むリサーチ
- `@Bot academic: <テーマ>` - ArXiv論文を含む学術リサーチ
- `@Bot paper: <ArXiv ID>` - 特定の論文を詳細分析
- `@Bot save: <コマンド>` - 結果をファイルとして保存
- `@Bot help` - ヘルプメッセージを表示

### 使用例

```
@Bot academic: 強化学習の最新動向
@Bot paper: 2501.01433
@Bot save: academic: マルチエージェント
```

## 主な機能の詳細

### 1. 基本リサーチ (`basic:`)
LLMの知識のみを使用して、指定されたトピックに関する基本的な情報を提供します。外部検索は行いません。

### 2. Web検索リサーチ (`web:`)
Google検索APIを使用して最新の情報を収集し、指定されたトピックに関する包括的なレポートを作成します。

### 3. 学術リサーチ (`academic:`)
ArXiv論文検索とGoogle検索を組み合わせ、学術的な分析を含む詳細なリサーチレポートを作成します。最新の論文情報と研究動向を把握できます。

### 4. 論文分析 (`paper:`)
特定のArXiv論文IDを指定して、その論文の詳細な分析を行います。研究の目的、方法論、結果、意義などを日本語で要約します。

## アーキテクチャ

このBotは以下の技術スタックで構築されています：

- **Discord.py**: DiscordとのインターフェースとBot機能
- **LangChain**: プロンプトの管理と処理パイプラインの構築
- **Google Gemini 2.0**: 自然言語処理と論文分析
- **ArXiv API**: 学術論文の検索と取得
- **Google Custom Search API**: Web検索機能（オプション）

## 制限事項

- 長い研究結果はDiscordのメッセージ長制限により分割されます
- 検索結果の品質はクエリの質とAPIの可用性に依存します
- ArXiv APIには利用制限があるため、大量の検索リクエストを短時間に行うと制限される場合があります

## ライセンス

MIT License

## 貢献

Pull requestsやIssuesを歓迎します。大きな変更を加える前には、まずIssueを開いて議論してください。

---

## 開発者向け情報

### 環境変数の詳細

- `GOOGLE_API_KEY`: Gemini APIを使用するためのAPIキー
- `GOOGLE_CSE_ID`: Google Custom Search Engineのカスタム検索エンジンID 
- `DISCORD_BOT_TOKEN`: Discord Developer Portalで取得したBotトークン
