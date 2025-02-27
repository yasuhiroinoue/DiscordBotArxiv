"""
Gemini Deep Research Tool with ArXiv & bioRxiv Integration - Discord Bot版

このツールは、Gemini LLMを使用して学術研究をサポートします。
ArXiv論文検索、bioRxiv論文検索、およびGoogle検索を統合して包括的な研究レポートを生成します。
Discordからの入力を受け取り、結果をDiscordに送信します。
"""

import os
import time
import re
import requests
import io
import discord
from discord.ext import commands
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")  # Discord Bot Token

# 必要なライブラリのインポート
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable

# LangChain関連のインポート
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_community import GoogleSearchAPIWrapper

# Discordの設定
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=None, intents=intents)

# Discord用定数
MAX_DISCORD_LENGTH = 1900  # 余裕を持たせて1900文字に設定

# バックグラウンド処理用のスレッドプール
executor = ThreadPoolExecutor(max_workers=2)

# 設定とモデル
@dataclass
class Config:
    """アプリケーション設定を管理するクラス"""
    # API設定
    google_api_key: str = google_api_key
    google_cse_id: str = google_cse_id

    # モデル設定
    llm_model: str = "gemini-2.0-flash"
    embedding_model: str = "models/embedding-001"
    
    # 検索設定
    google_search_results: int = 10
    arxiv_max_papers: int = 100
    biorxiv_max_papers: int = 100
    arxiv_api_delay: int = 1
    keyword_limit: int = 20
    arxiv_research_max_papers: int = 6
    biorxiv_research_max_papers: int = 6
    
    # テキスト処理設定
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_content_length: int = 10000

# 設定の初期化
config = Config()

# 環境変数の設定
os.environ["GOOGLE_API_KEY"] = config.google_api_key
os.environ["GOOGLE_CSE_ID"] = config.google_cse_id

# ===== ユーティリティ関数 =====

# 安全にメッセージを送信するヘルパー関数
async def safe_send(channel, content):
    """
    Discordのメッセージ長制限を考慮して安全にメッセージを送信する
    """
    if not content:
        return
        
    if len(content) <= MAX_DISCORD_LENGTH:
        await channel.send(content)
    else:
        # メッセージが長すぎる場合は分割して送信
        await split_and_send_messages(channel, content)

# エラーメッセージを安全に送信
async def send_error(channel, error):
    """
    エラーメッセージを安全に送信する
    長すぎるエラーメッセージは適切に短縮する
    """
    error_str = str(error)
    # エラーメッセージが長すぎる場合は短縮
    if len(error_str) > MAX_DISCORD_LENGTH:
        error_str = error_str[:MAX_DISCORD_LENGTH - 100] + "...(省略)"
    
    await channel.send(f"❌ エラーが発生しました: {error_str}")

# モデルとツールの初期化
def init_models_and_tools():
    """LLMモデルと検索ツールを初期化する"""
    # Gemini LLMモデルの初期化
    llm = ChatGoogleGenerativeAI(model=config.llm_model)
    embeddings = GoogleGenerativeAIEmbeddings(model=config.embedding_model)
    
    # テキスト分割機能の初期化
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    
    # Google検索の初期化
    google_search_available = False
    google_search_tool = None
    
    try:
        search = GoogleSearchAPIWrapper(k=config.google_search_results)
        google_search_tool = Tool(
            name="google_search",
            description="Search Google for recent results.",
            func=search.run
        )
        google_search_available = True
        print("Google検索が正常に初期化されました。")
    except Exception as e:
        print(f"Google検索の初期化に失敗しました: {e}")
    
    return llm, embeddings, text_splitter, google_search_tool, google_search_available

# モデルとツールの初期化
llm, embeddings, text_splitter, google_search_tool, google_search_available = init_models_and_tools()

def check_arxiv_available() -> bool:
    """arxivパッケージが利用可能かを確認する"""
    try:
        import arxiv
        return True
    except ImportError:
        print("arxivパッケージがインストールされていません。")
        print("pip install arxiv を実行してインストールしてください。")
        return False

def format_papers_data(papers: List[Dict[str, Any]]) -> str:
    """論文データを文字列形式に整形する (ArXivとbioRxiv両方に対応)"""
    if not papers:
        return ""
        
    papers_data = "## 論文データ:\n"
    for i, paper in enumerate(papers):
        source = paper.get("source", "arxiv")  # ソースを取得、デフォルトはarxiv
        papers_data += f"論文{i+1} [{source}]: {paper['title']}\n"
        papers_data += f"著者: {paper['authors']}\n"
        papers_data += f"公開日: {paper['published']}\n"
        papers_data += f"URL: {paper['url']}\n"
        papers_data += f"要約: {paper['summary']}\n"
        papers_data += f"詳細: {paper.get('detailed_summary', '要約なし')}\n\n"
    
    return papers_data

def highlight_references(report: str) -> str:
    """参考文献セクションを強調表示し、URLをハイパーリンクに変換する"""
    if "## 参考文献" not in report:
        return report
        
    report_parts = report.split("## 参考文献")
    references_section = "## 参考文献" + report_parts[1]
    
    # 参考文献セクションのタイトルを変更
    highlighted_references = references_section.replace(
        "## 参考文献", 
        "## 参考文献（実際に使用された情報源）"
    )
    
    # URLをハイパーリンクに変換
    # URLパターンにマッチする正規表現
    url_pattern = r'(https?://\S+)'
    
    # 行ごとに処理
    lines = highlighted_references.split('\n')
    for i, line in enumerate(lines):
        # URLを検出
        urls = re.findall(url_pattern, line)
        for url in urls:
            clean_url = url.rstrip('.,;:)')  # URLの末尾の記号を除去
            # URLを<URL>形式に置換（Discordでクリック可能になる）
            line = line.replace(url, f'<{clean_url}>')
        lines[i] = line
    
    highlighted_references = '\n'.join(lines)
    
    return report_parts[0] + highlighted_references

# ===== Discord用の共通メソッド =====

async def split_and_send_messages(channel, text, max_length=MAX_DISCORD_LENGTH):
    """長文のメッセージを適切に分割して送信する"""
    if not text:
        await channel.send("生成された結果がありません。")
        return
        
    # 非常に長いテキストの場合はファイルとして送信
    if len(text) > MAX_DISCORD_LENGTH * 10:
        await channel.send("結果が非常に長いため、ファイルとして送信します...")
        return await save_response_as_file(channel, text, "long_research_result.md")
    
    start = 0
    part_num = 1
    total_parts = (len(text) + max_length - 1) // max_length  # 切り上げ除算
    
    while start < len(text):
        end = start + max_length
        
        # 適切な区切り位置を探す（改行や空白が望ましい）
        if end < len(text):
            # 後方から適切な区切り位置を探す
            pos = end
            while pos > start + max_length // 2:
                if text[pos] in ' \n\r\t':
                    end = pos
                    break
                pos -= 1
                
            # 適切な区切り位置が見つからなかった場合は強制的に区切る
            if pos == start + max_length // 2:
                end = start + max_length
                
        part_message = text[start:end].strip()
        
        try:
            if total_parts > 1:
                await channel.send(f"**パート {part_num}/{total_parts}**\n{part_message}")
            else:
                await channel.send(part_message)
        except discord.HTTPException as e:
            # 送信エラーが発生した場合
            await channel.send(f"メッセージの送信中にエラーが発生しました。結果をファイルとして保存します。")
            return await save_response_as_file(channel, text, "error_research_result.md")
            
        start = end
        part_num += 1
        
        # 大量のメッセージを送信する場合はレート制限を避けるために少し待機
        if total_parts > 3:
            await asyncio.sleep(1)

async def save_response_as_file(channel, response_text, filename=None):
    """レスポンスをファイルとして保存して送信する"""
    if filename is None:
        filename = "research_report.md"
    
    try:
        # ファイルオブジェクトを作成
        file = discord.File(io.StringIO(response_text), filename=filename)
        
        # ファイルを添付したメッセージを送信
        await channel.send(f"📊 レポートをファイルとして保存しました:", file=file)
    except Exception as e:
        # ファイル送信に失敗した場合
        error_msg = f"ファイルの送信に失敗しました: {str(e)[:100]}..."
        await channel.send(error_msg)
        
        # 内容を要約して送信
        if len(response_text) > 500:
            summary = response_text[:500] + "...(省略)"
            await channel.send(f"レポートの一部:\n```\n{summary}\n```")

# ===== 検索機能 =====

def web_search(query: str) -> str:
    """Google検索を実行し結果を返す"""
    print(f"\n---Google検索の実行: '{query}'---")
    
    if not google_search_available:
        print("Google検索が利用できません。この部分はスキップします。")
        return "Google検索は現在利用できません。"
    
    try:
        results = google_search_tool.run(query)
        print("検索結果を取得しました。")
        return results
    except Exception as e:
        print(f"Google検索エラー: {e}")
        return f"検索エラーが発生しました: {str(e)}"

def arxiv_research(query: str, max_papers: int = None) -> Dict[str, Any]:
    """ArXivから論文を検索して処理する"""
    if max_papers is None:
        max_papers = config.arxiv_max_papers
        
    print(f"\n------ ArXivで「{query}」に関する論文を検索中... ------")
    
    if not check_arxiv_available():
        return {"status": "error", "message": "arxivパッケージがインストールされていません", "papers": []}
    
    try:
        import arxiv
        
        # arxivクライアントを直接使用
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_papers,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = list(client.results(search))
        print(f"ArXiv検索結果: {len(papers)}件の論文が見つかりました。")
        
        if not papers:
            print("ArXiv検索: 該当する論文が見つかりませんでした。")
            return {"status": "error", "message": "該当する論文が見つかりませんでした。", "papers": []}
        
        # 検索結果を整形
        paper_results = []
        for i, paper in enumerate(papers):
            paper_url = f"https://arxiv.org/abs/{paper.get_short_id()}"
            print(f"論文{i+1}: {paper.title} - {paper_url}")
            
            # 本文を取得
            content = f"タイトル: {paper.title}\n要約: {paper.summary}"
            
            paper_info = {
                "title": paper.title,
                "authors": ", ".join(author.name for author in paper.authors),
                "published": paper.published.strftime("%Y-%m-%d"),
                "summary": paper.summary,
                "url": paper_url,
                "content": content,
                "source": "arxiv"  # ソース識別用
            }
            paper_results.append(paper_info)
            if i < len(papers) - 1:
                time.sleep(config.arxiv_api_delay)  # API制限に配慮
        
        return {
            "status": "success", 
            "papers": paper_results,
            "count": len(paper_results)
        }
        
    except Exception as e:
        print(f"ArXiv検索処理エラー: {e}")
        return {"status": "error", "message": f"ArXiv検索エラー: {str(e)}", "papers": []}

def get_biorxiv_abstract(doi: str) -> str:
    """bioRxiv APIを使用して論文の要約を取得する"""
    try:
        # DOIからbioRxiv APIを使用
        api_url = f"https://api.biorxiv.org/details/biorxiv/{doi}"
        response = requests.get(api_url)
        
        if response.status_code == 200:
            data = response.json()
            collection = data.get("collection", [])
            if collection:
                abstract = collection[0].get("abstract", "")
                if abstract:
                    return abstract
        
        # APIで取得できない場合はWebスクレイピングで試行
        paper_url = f"https://www.biorxiv.org/content/{doi}v1"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(paper_url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            abstract_elem = soup.find('div', class_='abstract')
            if abstract_elem:
                return abstract_elem.text.strip().replace('Abstract', '', 1).strip()
    
    except Exception as e:
        print(f"要約取得エラー (DOI {doi}): {e}")
    
    return "要約なし"

def fallback_to_web_search(query: str, max_papers: int) -> Dict[str, Any]:
    """PubMed検索が失敗した場合のWebスクレイピングによるフォールバック"""
    print(f"bioRxiv Webサイト直接検索にフォールバックします...")
    
    try:
        # bioRxiv検索ページ
        search_url = f"https://www.biorxiv.org/search/{query}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(search_url, headers=headers)
        
        if response.status_code != 200:
            print(f"bioRxiv Web検索エラー: ステータスコード {response.status_code}")
            return {"status": "error", "message": f"bioRxiv Web検索エラー: {response.status_code}", "papers": []}
        
        # HTMLの解析
        soup = BeautifulSoup(response.text, 'html.parser')
        papers = soup.find_all('li', class_='search-result')
        
        if not papers:
            print("bioRxiv検索: 該当する論文が見つかりませんでした。")
            return {"status": "error", "message": "該当する論文が見つかりませんでした。", "papers": []}
        
        # 検索結果を整形
        paper_results = []
        count = 0
        
        for paper in papers:
            if count >= max_papers:
                break
                
            title_elem = paper.find('span', class_='highwire-cite-title')
            if not title_elem:
                continue
                
            title = title_elem.text.strip()
            
            # 著者情報
            authors_elem = paper.find('span', class_='highwire-citation-authors')
            authors = authors_elem.text.strip() if authors_elem else '著者不明'
            
            # URL情報
            link_elem = title_elem.find('a')
            paper_url = f"https://www.biorxiv.org{link_elem['href']}" if link_elem and 'href' in link_elem.attrs else ""
            
            # DOI情報を抽出
            doi = ""
            if paper_url:
                doi_match = re.search(r'10\.1101/(.+?)($|/)', paper_url)
                if doi_match:
                    doi = "10.1101/" + doi_match.group(1)
            
            # 要約（アブストラクト）の取得
            abstract = ""
            if paper_url:
                try:
                    paper_response = requests.get(paper_url, headers=headers)
                    if paper_response.status_code == 200:
                        paper_soup = BeautifulSoup(paper_response.text, 'html.parser')
                        abstract_elem = paper_soup.find('div', class_='abstract')
                        if abstract_elem:
                            abstract = abstract_elem.text.strip().replace('Abstract', '', 1).strip()
                except Exception as e:
                    print(f"論文詳細取得エラー: {e}")
            
            # 日付情報
            date_elem = paper.find('span', class_='highwire-cite-metadata-date')
            date = date_elem.text.strip() if date_elem else '日付不明'
            
            print(f"論文{count+1}: {title} - {paper_url}")
            
            # 論文情報を整形
            content = f"タイトル: {title}\n要約: {abstract}"
            
            paper_info = {
                "title": title,
                "authors": authors,
                "published": date,
                "summary": abstract,
                "url": paper_url,
                "doi": doi,
                "content": content,
                "source": "biorxiv"
            }
            paper_results.append(paper_info)
            count += 1
            
            if count < max_papers:
                time.sleep(config.arxiv_api_delay)  # 過負荷防止
        
        return {
            "status": "success", 
            "papers": paper_results,
            "count": len(paper_results)
        }
        
    except Exception as e:
        print(f"bioRxiv Web検索処理エラー: {e}")
        return {"status": "error", "message": f"bioRxiv Web検索エラー: {str(e)}", "papers": []}

def biorxiv_search(query: str, max_papers: int = None) -> Dict[str, Any]:
    """PubMed APIを使用してbioRxiv論文をキーワード検索する"""
    if max_papers is None:
        max_papers = config.biorxiv_max_papers
        
    print(f"\n------ bioRxivで「{query}」に関する論文を検索中 (PubMed API経由)... ------")
    
    try:
        # bioRxivに絞ったPubMed検索クエリを構築
        pubmed_query = f'{query} AND "bioRxiv"[Journal]'
        
        # PubMed検索を実行
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": pubmed_query,
            "retmode": "json",
            "retmax": max_papers
        }
        search_response = requests.get(search_url, params=search_params)
        
        if search_response.status_code != 200:
            print(f"PubMed検索エラー: ステータスコード {search_response.status_code}")
            return fallback_to_web_search(query, max_papers)
        
        search_results = search_response.json()
        id_list = search_results.get("esearchresult", {}).get("idlist", [])
        
        if not id_list:
            print("PubMed検索: 該当する論文が見つかりませんでした。Webスクレイピングにフォールバックします。")
            return fallback_to_web_search(query, max_papers)
            
        print(f"PubMed検索結果: {len(id_list)}件のbioRxiv論文が見つかりました")
        
        # 論文詳細情報を取得
        details_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        details_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "json"
        }
        details_response = requests.get(details_url, params=details_params)
        
        if details_response.status_code != 200:
            print(f"PubMed詳細取得エラー: ステータスコード {details_response.status_code}")
            return fallback_to_web_search(query, max_papers)
            
        details_data = details_response.json()
        
        # 結果を整形
        paper_results = []
        for pmid in id_list:
            try:
                paper_data = details_data["result"][pmid]
                
                # 論文情報を抽出
                title = paper_data.get("title", "タイトル不明")
                authors_list = paper_data.get("authors", [])
                authors = ", ".join([author.get("name", "") for author in authors_list if author.get("name")])
                
                # DOIの抽出
                article_ids = paper_data.get("articleids", [])
                doi = ""
                for id_obj in article_ids:
                    if id_obj.get("idtype") == "doi":
                        doi = id_obj.get("value", "")
                        break
                
                # bioRxiv DOIの形式チェック
                if doi and doi.startswith("10.1101/"):
                    # DOIからbioRxiv URLを構築
                    paper_url = f"https://doi.org/{doi}"
                    
                    # 要約はPubMedから直接取得できないので、bioRxiv APIで取得を試みる
                    abstract = get_biorxiv_abstract(doi)
                    
                    published_date = paper_data.get("pubdate", "日付不明")
                    
                    print(f"論文: {title} - {paper_url}")
                    
                    # 論文情報を整形
                    content = f"タイトル: {title}\n要約: {abstract}"
                    
                    paper_info = {
                        "title": title,
                        "authors": authors,
                        "published": published_date,
                        "summary": abstract,
                        "url": paper_url,
                        "doi": doi,
                        "content": content,
                        "source": "biorxiv"
                    }
                    paper_results.append(paper_info)
            except Exception as e:
                print(f"論文データ処理エラー (PMID {pmid}): {e}")
        
        if not paper_results:
            print("有効なbioRxiv論文が見つかりませんでした。Webスクレイピングにフォールバックします。")
            return fallback_to_web_search(query, max_papers)
            
        return {
            "status": "success", 
            "papers": paper_results,
            "count": len(paper_results)
        }
        
    except Exception as e:
        print(f"PubMed検索処理エラー: {e}")
        return fallback_to_web_search(query, max_papers)

def optimized_arxiv_search(topic: str, max_papers: int = None) -> Dict[str, Any]:
    """検索クエリを最適化してArXiv検索を実行する"""
    if max_papers is None:
        max_papers = config.arxiv_max_papers
        
    if not check_arxiv_available():
        return {"status": "error", "message": "arxivパッケージがインストールされていません", "papers": []}
    
    try:
        # 検索クエリの最適化
        query_optimization_prompt = PromptTemplate(
            input_variables=["topic"],
            template="""
            以下の研究テーマに対するArXiv検索クエリを最適化してください。
            できるだけ関連性の高い論文が検索できるよう、適切な検索構文を使用してください。
            英語で出力してください。検索クエリだけを出力し、余分な説明は不要です。
            
            研究テーマ: {topic}
            """
        )
        
        # Runnable APIを使用
        query_chain = query_optimization_prompt | llm | StrOutputParser()
        optimized_query = query_chain.invoke({"topic": topic}).strip()
        
        print(f"最適化されたArXivクエリ: {optimized_query}")
        
        # 最適化されたクエリで検索
        result = arxiv_research(optimized_query, max_papers=max_papers)
        
        # 結果がない場合は別のアプローチで再試行
        if result["status"] == "error" or len(result.get("papers", [])) == 0:
            print("最適化クエリで結果が得られませんでした。基本クエリで再試行します。")
            # 英語に翻訳して再試行
            translation_prompt = PromptTemplate(
                input_variables=["topic"],
                template="以下の研究テーマを英語に翻訳してください。翻訳だけを出力し、余分な説明は不要です。\n\n研究テーマ: {topic}"
            )
            translation_chain = translation_prompt | llm | StrOutputParser()
            english_topic = translation_chain.invoke({"topic": topic}).strip()
            
            print(f"英語トピックに変換: '{english_topic}'")
            result = arxiv_research(english_topic, max_papers=max_papers)
        
        return result
    
    except Exception as e:
        print(f"検索最適化エラー: {e}")
        return {"status": "error", "message": f"検索最適化エラー: {str(e)}", "papers": []}

def optimized_biorxiv_search(topic: str, max_papers: int = None) -> Dict[str, Any]:
    """検索クエリを最適化してbioRxiv検索を実行する"""
    if max_papers is None:
        max_papers = config.biorxiv_max_papers
    
    try:
        # 検索クエリの最適化
        query_optimization_prompt = PromptTemplate(
            input_variables=["topic"],
            template="""
            以下の研究テーマに対するPubMed検索クエリを最適化してください。
            生物学・医学分野のプレプリントサーバーbioRxivでの検索に適したキーワードにしてください。
            英語で出力してください。検索クエリだけを出力し、余分な説明は不要です。
            
            PubMedとbioRxiv検索用のクエリを作成します。
            
            研究テーマ: {topic}
            """
        )
        
        # Runnable APIを使用
        query_chain = query_optimization_prompt | llm | StrOutputParser()
        optimized_query = query_chain.invoke({"topic": topic}).strip()
        
        print(f"最適化されたbioRxivクエリ: {optimized_query}")
        
        # 最適化されたクエリで検索
        result = biorxiv_search(optimized_query, max_papers=max_papers)
        
        # 結果がない場合は別のアプローチで再試行
        if result["status"] == "error" or len(result.get("papers", [])) == 0:
            print("最適化クエリで結果が得られませんでした。基本クエリで再試行します。")
            # 英語に翻訳して再試行
            translation_prompt = PromptTemplate(
                input_variables=["topic"],
                template="以下の研究テーマを英語に翻訳してください。生物学用語がある場合は、適切な英語の専門用語を使用してください。翻訳だけを出力し、余分な説明は不要です。\n\n研究テーマ: {topic}"
            )
            translation_chain = translation_prompt | llm | StrOutputParser()
            english_topic = translation_chain.invoke({"topic": topic}).strip()
            
            print(f"英語トピックに変換: '{english_topic}'")
            result = biorxiv_search(english_topic, max_papers=max_papers)
        
        return result
    
    except Exception as e:
        print(f"検索最適化エラー: {e}")
        return {"status": "error", "message": f"検索最適化エラー: {str(e)}", "papers": []}

# ===== コンテンツ処理機能 =====

def summarize_paper(paper_content: str) -> str:
    """論文の内容を要約する"""
    if not paper_content or len(paper_content) < 100:
        return "十分な内容がありません。"
    
    # 要約用プロンプトテンプレート
    prompt_template = PromptTemplate(
        template="""
        以下は学術論文の内容です。この論文の主要なポイント、方法論、結果、および意義を簡潔に要約してください。
        専門用語は保持しつつ、理解しやすい日本語で説明してください。
        
        論文内容:
        {text}
        
        要約:
        """,
        input_variables=["text"]
    )
    
    try:
        # テキストが長すぎる場合は分割して処理
        if len(paper_content) > config.max_content_length:
            paper_content = paper_content[:config.max_content_length] + "..."
        
        # Runnable APIを使用
        summary_chain = prompt_template | llm | StrOutputParser()
        summary = summary_chain.invoke({"text": paper_content})
        return summary
    except Exception as e:
        print(f"要約生成エラー: {e}")
        return f"要約生成エラー: {str(e)}"

def analyze_paper(paper_url: str) -> str:
    """論文の詳細分析を行う"""
    try:
        print(f"\n論文分析を開始: {paper_url}")
        
        # ArXiv論文かbioRxiv論文かを判断
        if "arxiv" in paper_url.lower():
            return analyze_arxiv_paper(paper_url)
        elif "biorxiv" in paper_url.lower() or "doi.org" in paper_url.lower():
            return analyze_biorxiv_paper(paper_url)
        else:
            # URLからどのタイプか判断できない場合
            if paper_url.startswith("10.1101/"):
                # DOI形式でbioRxivと判断
                return analyze_biorxiv_paper(paper_url)
            else:
                # デフォルトはArXivとして扱う
                return analyze_arxiv_paper(paper_url)
    
    except Exception as e:
        print(f"論文分析エラー: {e}")
        return f"論文分析エラー: {str(e)}"

def analyze_arxiv_paper(paper_url: str) -> str:
    """ArXiv論文の詳細分析を行う"""
    try:
        print(f"\nArXiv論文分析を開始: {paper_url}")
        paper_id = paper_url.split("/")[-1]
        
        if not check_arxiv_available():
            return "arxivパッケージがインストールされていません。pip install arxiv を実行してインストールしてください。"
        
        import arxiv
        
        # 特定の論文IDで検索
        client = arxiv.Client()
        search = arxiv.Search(
            id_list=[paper_id],
            max_results=1
        )
        
        papers = list(client.results(search))
        
        if not papers:
            return "論文が見つかりませんでした。"
        
        paper = papers[0]
        paper_url = f"https://arxiv.org/abs/{paper.get_short_id()}"
        print(f"論文を取得しました: {paper.title}")
        
        # 論文の詳細分析用プロンプト
        analysis_prompt = PromptTemplate(
            input_variables=["title", "authors", "abstract", "url"],
            template="""
            以下の論文について詳細な分析を行ってください:
            
            タイトル: {title}
            著者: {authors}
            アブストラクト: {abstract}
            論文URL: {url}
            
            以下の構造で分析を行ってください:
            
            # 論文「{title}」の分析
            
            ## 研究の主な目的
            [論文の目的、背景、研究課題の説明]
            
            ## 使用されている方法論
            [著者が採用した研究手法、モデル、データセットなどの説明]
            
            ## 主要な結果と発見
            [論文の重要な結果、発見、数値などの要約]
            
            ## 研究の意義と影響
            [本研究の学術的・実用的な重要性の分析]
            
            ## 制限事項と今後の研究方向
            [著者が認識している制限事項と将来の研究機会]
            
            ## 参考文献情報
            論文情報: {title} - {authors} - {url}
            """
        )
        
        # Runnable APIを使用
        analysis_chain = analysis_prompt | llm | StrOutputParser()
        analysis = analysis_chain.invoke({
            "title": paper.title,
            "authors": ", ".join(author.name for author in paper.authors),
            "abstract": paper.summary,
            "url": paper_url
        })
        
        return analysis
    
    except Exception as e:
        print(f"ArXiv論文分析エラー: {e}")
        return f"ArXiv論文分析エラー: {str(e)}"

def analyze_biorxiv_paper(paper_doi: str) -> str:
    """bioRxiv論文の詳細分析を行う"""
    try:
        print(f"\nbioRxiv論文分析を開始: {paper_doi}")
        
        # DOIの抽出（URLから）
        if "doi.org/" in paper_doi:
            paper_doi = paper_doi.split("doi.org/")[-1]
        
        # bioRxiv DOIの形式（10.1101/...）確認
        if not paper_doi.startswith("10.1101/"):
            paper_doi = "10.1101/" + paper_doi
        
        # DOIから論文情報を取得
        url = f"https://api.biorxiv.org/details/biorxiv/{paper_doi}"
        response = requests.get(url)
        
        if response.status_code != 200:
            return f"bioRxiv API エラー: ステータスコード {response.status_code}"
        
        data = response.json()
        collection = data.get("collection", [])
        
        if not collection:
            # API取得に失敗した場合はWebスクレイピングを試みる
            print("API経由での論文情報取得に失敗しました。Webスクレイピングを試みます。")
            try:
                paper_url = f"https://doi.org/{paper_doi}"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(paper_url, headers=headers)
                
                if response.status_code != 200:
                    return "論文が見つかりませんでした。"
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # タイトル取得
                title_elem = soup.find('meta', {'name': 'citation_title'})
                title = title_elem['content'] if title_elem else "タイトル不明"
                
                # 著者取得
                author_elems = soup.find_all('meta', {'name': 'citation_author'})
                authors = ", ".join([elem['content'] for elem in author_elems]) if author_elems else "著者不明"
                
                # 要約取得
                abstract_elem = soup.find('div', class_='abstract')
                abstract = abstract_elem.text.strip().replace('Abstract', '', 1).strip() if abstract_elem else "要約なし"
                
            except Exception as e:
                print(f"Webスクレイピングエラー: {e}")
                return "論文が見つかりませんでした。"
        else:
            paper = collection[0]
            title = paper.get('title', '無題')
            authors = paper.get('authors', '著者不明')
            abstract = paper.get('abstract', '要約なし')
            paper_url = f"https://doi.org/{paper_doi}"
        
        print(f"論文を取得しました: {title}")
        
        # 論文の詳細分析用プロンプト
        analysis_prompt = PromptTemplate(
            input_variables=["title", "authors", "abstract", "url"],
            template="""
            以下のbioRxiv論文について詳細な分析を行ってください:
            
            タイトル: {title}
            著者: {authors}
            アブストラクト: {abstract}
            論文URL: {url}
            
            以下の構造で分析を行ってください:
            
            # 論文「{title}」の分析
            
            ## 研究の主な目的
            [論文の目的、背景、研究課題の説明]
            
            ## 使用されている方法論
            [著者が採用した研究手法、モデル、データセットなどの説明]
            
            ## 主要な結果と発見
            [論文の重要な結果、発見、数値などの要約]
            
            ## 研究の意義と影響
            [本研究の学術的・実用的な重要性の分析]
            
            ## 制限事項と今後の研究方向
            [著者が認識している制限事項と将来の研究機会]
            
            ## 参考文献情報
            論文情報: {title} - {authors} - {url}
            """
        )
        
        # Runnable APIを使用
        analysis_chain = analysis_prompt | llm | StrOutputParser()
        analysis = analysis_chain.invoke({
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "url": paper_url
        })
        
        return analysis
    
    except Exception as e:
        print(f"bioRxiv論文分析エラー: {e}")
        return f"bioRxiv論文分析エラー: {str(e)}"

# ===== リサーチ機能 =====

def academic_research(topic: str) -> str:
    """学術リサーチを実行する総合機能（ArXivとbioRxivを含む）"""
    try:
        print(f"\n=== '{topic}'に関する学術リサーチを開始 ===")
        
        # テーマに関連するキーワードを抽出
        keyword_prompt = PromptTemplate(
            input_variables=["topic"],
            template="""
            以下の研究テーマに関する効果的な学術検索のためのキーワードを3つ生成してください。
            各キーワードは引用符で囲まれた形で提供してください。英語で出力してください。
            
            研究テーマ: {topic}
            """
        )
        
        # キーワード抽出
        keyword_chain = keyword_prompt | llm | StrOutputParser()
        keywords_text = keyword_chain.invoke({"topic": topic}).strip()
        
        # キーワードを抽出（引用符で囲まれた部分を取得）
        keywords = re.findall(r'"([^"]*)"', keywords_text)
        if not keywords:
            keywords = [topic]  # キーワード抽出に失敗した場合はトピックをそのまま使用
        
        print(f"抽出されたキーワード: {keywords}")
        
        # ArXiv検索とbioRxiv検索を実行
        arxiv_results = []
        biorxiv_results = []
        all_papers = []
        
        arxiv_available = check_arxiv_available()
        
        # 最初のN個のキーワードのみ使用して負荷軽減
        for keyword in keywords[:config.keyword_limit]:  
            # ArXiv検索
            if arxiv_available:
                result = optimized_arxiv_search(keyword, max_papers=config.arxiv_research_max_papers)
                if result["status"] == "success":
                    arxiv_results.extend(result["papers"])
                    
            # bioRxiv検索（英語キーワードのみ使用）
            biorxiv_result = optimized_biorxiv_search(keyword, max_papers=config.biorxiv_research_max_papers)
            if biorxiv_result["status"] == "success":
                biorxiv_results.extend(biorxiv_result["papers"])
                
            time.sleep(2)  # API制限に配慮
        
        # すべての論文を結合
        all_papers = arxiv_results + biorxiv_results
        
        # 重複論文を除去
        unique_papers = {}
        for paper in all_papers:
            if paper["url"] not in unique_papers:
                unique_papers[paper["url"]] = paper
        
        all_papers = list(unique_papers.values())
        
        # 検索結果の表示
        print(f"\n=== 学術検索結果: 合計{len(all_papers)}件 (ArXiv: {len(arxiv_results)}件, bioRxiv: {len(biorxiv_results)}件) ===")
        for i, paper in enumerate(all_papers):
            source = paper.get("source", "arxiv")  # ソースを取得、デフォルトはarxiv
            print(f"{i+1}. [{source}] {paper['title']} - {paper['url']}")
        
        # 論文内容の要約
        for paper in all_papers:
            print(f"論文を要約中: {paper['title']}")
            paper["detailed_summary"] = summarize_paper(paper["content"])
        
        # Web検索を実行（Google検索が使用可能な場合）
        web_search_results = ""
        if google_search_available:
            print("Google検索を実行中...")
            web_search_results = web_search(topic)
            print("Google検索完了")
        
        # 論文データを文字列形式に整形
        all_papers_data = format_papers_data(all_papers)
        
        # Web検索データを文字列形式に整形
        web_search_data = ""
        if web_search_results:
            web_search_data = "## Web検索データ:\n" + web_search_results
        
        # 統合レポートの作成
        integration_prompt = PromptTemplate(
            input_variables=["topic", "papers_data", "web_search_data", "has_papers", "has_web"],
            template="""
            以下の研究テーマに関する包括的な学術レポートを作成してください:
            
            研究テーマ: {topic}
            
            {papers_data}
            
            {web_search_data}
            
            以下の形式で学術研究レポートを作成してください:
            
            # {topic}に関する学術研究レポート
            
            ## 概要
            [テーマの概要と重要性]
            
            ## 研究の現状
            [現在の研究状況と主要なアプローチ]
            
            ## 最新の研究動向
            [最近の論文や情報源から明らかになった新しい発見や方法論]
            
            ## 重要な発見と主要論文
            [分野における重要な発見とその根拠となる論文や情報源]
            
            ## 未解決の問題と将来の研究方向
            [現在の研究の限界と今後の方向性]
            
            ## 参考文献
            ここでは、上記で使用した情報源（ArXiv論文、bioRxiv論文、Web検索結果）を番号付きリストで提供してください。
            各参考文献には、タイトル、著者、URLとソース（ArXivかbioRxiv）を含めてください。
            例: [1] 論文タイトル - 著者名 - URL (ArXiv)
            
            注意: 参考文献は実際に上記で提供された情報に基づいてください。架空の参考文献は作成しないでください。
            """
        )
        
        # 統合レポート生成
        print("最終レポートを生成中...")
        integration_chain = integration_prompt | llm | StrOutputParser()
        final_report = integration_chain.invoke({
            "topic": topic,
            "papers_data": all_papers_data,
            "web_search_data": web_search_data,
            "has_papers": len(all_papers) > 0,
            "has_web": bool(web_search_results)
        })
        
        # 参考文献を強調表示
        highlighted_report = highlight_references(final_report)
        
        return highlighted_report
        
    except Exception as e:
        print(f"学術リサーチエラー: {e}")
        return f"学術リサーチ中にエラーが発生しました: {str(e)}"

def basic_research(topic: str) -> str:
    """LLMの知識のみを使用した基本リサーチを実行する"""
    try:
        # 基本的なプロンプトテンプレート
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="""
            あなたは研究テーマについて詳しく調査するリサーチアシスタントです。
            以下のトピックについて、あなたの知識を活用して詳細なレポートを作成してください。
            
            トピック: {topic}
            
            レポートは以下の構造で作成してください:
            # {topic}に関する基本リサーチレポート
            
            ## 概要
            [テーマの概要と重要性]
            
            ## 主要ポイント
            1. [ポイント1]
            2. [ポイント2]
            3. [ポイント3]
            
            ## 詳細分析
            [詳細な説明と考察]
            
            ## 結論と実践的応用
            [結論と実際の応用方法]
            
            注意: このレポートはLLMの知識に基づいており、外部検索は使用していません。
            """
        )
        
        # Runnable APIを使用
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"topic": topic})
        return response
        
    except Exception as e:
        print(f"基本リサーチエラー: {e}")
        return f"リサーチ処理中にエラーが発生しました: {str(e)}"

def web_research(topic: str) -> str:
    """Web検索を利用したリサーチを実行する"""
    try:
        print(f"\nWeb検索を活用した '{topic}' に関するリサーチを開始")
        
        if not google_search_available:
            print("Google検索が利用できません。基本リサーチに切り替えます。")
            return basic_research(topic)
        
        # Web検索実行
        search_results = web_search(topic)
        
        # 検索結果に基づいてレポート作成
        research_prompt = PromptTemplate(
            input_variables=["topic", "search_results"],
            template="""
            以下のWeb検索結果に基づいて、研究テーマに関する詳細なレポートを作成してください:
            
            研究テーマ: {topic}
            
            検索結果:
            {search_results}
            
            以下の形式でレポートを作成してください:
            
            # {topic}に関するWeb検索リサーチレポート
            
            ## 概要
            [テーマの概要と重要性]
            
            ## 主要な情報と発見事項
            [検索結果から得られた主要な情報をまとめる]
            
            ## 詳細分析
            [情報の詳細な分析と考察]
            
            ## 結論と応用
            [結論と実践的な応用可能性]
            
            ## 情報源
            [検索結果から得られた情報源]
            """
        )
        
        research_chain = research_prompt | llm | StrOutputParser()
        report = research_chain.invoke({
            "topic": topic,
            "search_results": search_results
        })
        
        return report
        
    except Exception as e:
        print(f"Web検索リサーチエラー: {e}")
        return f"Web検索処理中にエラーが発生しました: {str(e)}"

# ===== Discord Bot関数 =====

@bot.event
async def on_ready():
    """Botの起動時に呼び出される"""
    print("----------------------------------------")
    print(f'ArXiv & bioRxiv Research Bot Logged in as {bot.user}')
    print("----------------------------------------")

@bot.event
async def on_message(message):
    """メッセージ受信時に呼び出される"""
    # 自分自身のメッセージは無視
    if message.author == bot.user:
        return

    # Botへのメンションかダイレクトメッセージの場合
    if bot.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
        # メッセージ内容を取得
        content = message.content.replace(f'<@!{bot.user.id}>', '').replace(f'<@{bot.user.id}>', '').strip()
        
        # ヘルプコマンド
        if content.lower() == '!help':
            help_text = """
            **ArXiv & bioRxiv Research Bot - コマンド一覧**
            
            ボットの名前は`Bot`とします。
            `@Bot basic: <テーマ>` - LLMの知識のみを使用した基本リサーチ
            `@Bot web: <テーマ>` - Web検索を含むリサーチ
            `@Bot academic: <テーマ>` - ArXivとbioRxiv論文を含む学術リサーチ
            `@Bot arxiv: <テーマ>` - ArXiv論文のみを検索
            `@Bot biorxiv: <テーマ>` - bioRxiv論文のみを検索
            `@Bot paper: <ArXiv ID>` - ArXiv論文の詳細分析
            `@Bot biopaper: <DOI>` - bioRxiv論文の詳細分析
            `@Bot save: <コマンド>` - 結果をファイルとして保存
            `@Bot !help` - このヘルプメッセージを表示
            
            **例:**
            `@Bot academic: 強化学習の最新動向`
            `@Bot paper: 2501.01433`
            `@Bot biopaper: 10.1101/2022.01.01.123456`
            `@Bot biorxiv: CRISPR`
            `@Bot save: academic: マルチエージェント`
            """
            await message.channel.send(help_text)
            return

        # リサーチコマンドの処理
        save_as_file = False
        
        # ファイル保存オプションの処理
        if content.startswith('save:'):
            save_as_file = True
            content = content[5:].strip()
            
        # コマンドの処理
        if content.startswith('basic:'):
            topic = content[6:].strip()
            await message.add_reaction('🔍')
            
            # 進捗報告
            await safe_send(message.channel, f"「{topic}」の基本リサーチを開始しました...")
            
            # バックグラウンドでリサーチを実行
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(executor, basic_research, topic)
                
                if save_as_file:
                    await save_response_as_file(message.channel, result, f"basic_research_{topic[:20]}.md")
                else:
                    await split_and_send_messages(message.channel, result)
                
                await message.add_reaction('✅')
            except Exception as e:
                error_msg = f"エラーが発生しました: {str(e)[:200]}..." if len(str(e)) > 200 else f"エラーが発生しました: {str(e)}"
                await safe_send(message.channel, f"❌ {error_msg}")
                
        elif content.startswith('web:'):
            topic = content[4:].strip()
            await message.add_reaction('🌐')
            
            # 進捗報告
            await safe_send(message.channel, f"「{topic}」のWeb検索リサーチを開始しました...")
            
            # 定期的な進捗報告のための関数
            async def report_progress():
                progress_messages = [
                    "🔍 Web検索を実行中...",
                    "📊 検索結果を分析中...",
                    "✍️ リサーチレポートを作成中..."
                ]
                
                for i, msg in enumerate(progress_messages):
                    await asyncio.sleep(20)  # 20秒ごとに進捗報告
                    try:
                        await safe_send(message.channel, msg)
                    except:
                        pass  # 接続エラーなどを無視
            
            # 進捗報告タスクを開始
            progress_task = asyncio.create_task(report_progress())
            
            # バックグラウンドでリサーチを実行
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(executor, web_research, topic)
                
                # 進捗報告タスクをキャンセル
                progress_task.cancel()
                
                if save_as_file:
                    await save_response_as_file(message.channel, result, f"web_research_{topic[:20]}.md")
                else:
                    await split_and_send_messages(message.channel, result)
                
                await message.add_reaction('✅')
            except Exception as e:
                error_msg = f"エラーが発生しました: {str(e)[:200]}..." if len(str(e)) > 200 else f"エラーが発生しました: {str(e)}"
                await safe_send(message.channel, f"❌ {error_msg}")
                
        elif content.startswith('academic:'):
            topic = content[9:].strip()
            await message.add_reaction('📚')
            
            # 進捗報告
            await safe_send(message.channel, f"「{topic}」に関する学術リサーチを開始しました。完了までお待ちください...")
            
            # 定期的な進捗報告のための関数
            async def report_progress():
                progress_messages = [
                    "🔍 ArXivとbioRxiv論文を検索中...",
                    "📄 論文情報を解析中...",
                    "🌐 関連情報を収集中...",
                    "✍️ 研究レポートを作成中..."
                ]
                
                for i, msg in enumerate(progress_messages):
                    await asyncio.sleep(30)  # 30秒ごと
                    try:
                        await safe_send(message.channel, msg)
                    except:
                        pass  # 接続エラーなどを無視
            
            # 進捗報告タスクを開始
            progress_task = asyncio.create_task(report_progress())
            
            # バックグラウンドでリサーチを実行
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(executor, academic_research, topic)
                
                # 進捗報告タスクをキャンセル
                progress_task.cancel()
                
                if save_as_file:
                    await save_response_as_file(message.channel, result, f"academic_research_{topic[:20]}.md")
                else:
                    # 長い結果を分割して送信
                    await split_and_send_messages(message.channel, result)
                
                await message.add_reaction('✅')
            except Exception as e:
                # エラーメッセージが長すぎる場合は短縮
                tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                print(f"詳細なエラー:\n{tb_str}")
                error_msg = f"エラーが発生しました: {str(e)[:200]}..." if len(str(e)) > 200 else f"エラーが発生しました: {str(e)}"
                await safe_send(message.channel, f"❌ {error_msg}")
            
        elif content.startswith('arxiv:'):
            topic = content[6:].strip()
            await message.add_reaction('📜')
            
            # 進捗報告
            await safe_send(message.channel, f"「{topic}」に関するArXiv検索を開始しました...")
            
            # バックグラウンドで実行
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(executor, lambda: format_papers_data(optimized_arxiv_search(topic)["papers"]))
                
                if save_as_file:
                    await save_response_as_file(message.channel, result, f"arxiv_search_{topic[:20]}.md")
                else:
                    await split_and_send_messages(message.channel, result)
                
                await message.add_reaction('✅')
            except Exception as e:
                error_msg = f"エラーが発生しました: {str(e)[:200]}..." if len(str(e)) > 200 else f"エラーが発生しました: {str(e)}"
                await safe_send(message.channel, f"❌ {error_msg}")

        elif content.startswith('biorxiv:'):
            topic = content[8:].strip()
            await message.add_reaction('🧬')
            
            # 進捗報告
            await safe_send(message.channel, f"「{topic}」に関するbioRxiv検索を開始しました...")
            
            # バックグラウンドで実行
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(executor, lambda: format_papers_data(optimized_biorxiv_search(topic)["papers"]))
                
                if save_as_file:
                    await save_response_as_file(message.channel, result, f"biorxiv_search_{topic[:20]}.md")
                else:
                    await split_and_send_messages(message.channel, result)
                
                await message.add_reaction('✅')
            except Exception as e:
                error_msg = f"エラーが発生しました: {str(e)[:200]}..." if len(str(e)) > 200 else f"エラーが発生しました: {str(e)}"
                await safe_send(message.channel, f"❌ {error_msg}")
                
        elif content.startswith('paper:'):
            paper_id = content[6:].strip()
            await message.add_reaction('📄')
            
            # 論文IDを整形
            if not paper_id.startswith('http'):
                paper_id = f"https://arxiv.org/abs/{paper_id}"
            
            # 進捗報告
            await safe_send(message.channel, f"論文「{paper_id}」の分析を開始しました...")
            
            # バックグラウンドで実行
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(executor, analyze_arxiv_paper, paper_id)
                
                if save_as_file:
                    await save_response_as_file(message.channel, result, f"paper_analysis_{paper_id.split('/')[-1]}.md")
                else:
                    await split_and_send_messages(message.channel, result)
                
                await message.add_reaction('✅')
            except Exception as e:
                error_msg = f"エラーが発生しました: {str(e)[:200]}..." if len(str(e)) > 200 else f"エラーが発生しました: {str(e)}"
                await safe_send(message.channel, f"❌ {error_msg}")
        
        elif content.startswith('biopaper:'):
            doi = content[9:].strip()
            await message.add_reaction('🧬')
            
            # 進捗報告
            await safe_send(message.channel, f"bioRxiv論文「{doi}」の分析を開始しました...")
            
            # バックグラウンドで実行
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(executor, analyze_biorxiv_paper, doi)
                
                if save_as_file:
                    await save_response_as_file(message.channel, result, f"biorxiv_analysis_{doi.replace('/', '_')}.md")
                else:
                    await split_and_send_messages(message.channel, result)
                
                await message.add_reaction('✅')
            except Exception as e:
                error_msg = f"エラーが発生しました: {str(e)[:200]}..." if len(str(e)) > 200 else f"エラーが発生しました: {str(e)}"
                await safe_send(message.channel, f"❌ {error_msg}")
                
        else:
            # デフォルトは学術リサーチ
            topic = content
            await message.add_reaction('🔎')
            
            # 進捗報告
            await safe_send(message.channel, f"「{topic}」に関する学術リサーチを開始しました。完了までお待ちください...")
            
            # 定期的な進捗報告のための関数
            async def report_progress():
                progress_messages = [
                    "🔍 ArXivとbioRxiv論文を検索中...",
                    "📄 論文情報を解析中...",
                    "🌐 関連情報を収集中...",
                    "✍️ 研究レポートを作成中..."
                ]
                
                for i, msg in enumerate(progress_messages):
                    await asyncio.sleep(30)  # 30秒ごとに進捗報告
                    try:
                        await safe_send(message.channel, msg)
                    except:
                        pass  # 接続エラーなどを無視
            
            # 進捗報告タスクを開始
            progress_task = asyncio.create_task(report_progress())
            
            # バックグラウンドでリサーチを実行
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(executor, academic_research, topic)
                
                # 進捗報告タスクをキャンセル
                progress_task.cancel()
                
                if save_as_file:
                    await save_response_as_file(message.channel, result, f"research_{topic[:20]}.md")
                else:
                    await split_and_send_messages(message.channel, result)
                
                await message.add_reaction('✅')
            except Exception as e:
                error_msg = f"エラーが発生しました: {str(e)[:200]}..." if len(str(e)) > 200 else f"エラーが発生しました: {str(e)}"
                await safe_send(message.channel, f"❌ {error_msg}")

# Botの起動
if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
