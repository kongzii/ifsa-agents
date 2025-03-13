import os
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import requests
from dotenv import load_dotenv
from goose3 import Goose
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer, Probability
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.utils import utcnow

load_dotenv()

class YourAgent(DeployableTraderAgent):
    bet_on_n_markets_per_run = 1

    def load(self):
        self.goose = Goose()

    def extract_news_keywords(self, question: str) -> list[str]:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0.7,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an experienced news researcher helping to find the best keywords for a news search."),
            ("user", """
Think about the question and consider which keyword (phrases) could be useful for finding articles that are relevant for answering the question.
Keywords can be one word or a small keyword phrase.
Each of them will be used in a query to NewsAPI.
Return up to 5 keywords.

Question: {question}

Just return a comma-separated list of keywords, without any extra text.
            """)
        ])
        messages = prompt.format_messages(question=question)
        response = llm.invoke(messages, max_tokens=100).content
        keywords = [kw.strip() for kw in response.split(",") if kw.strip()]
        return keywords

    def fetch_articles_for_keyword(self, keyword: str, days_back: int = 7, page_size: int = 10) -> list[dict]:
        base_url = "https://newsapi.org/v2/everything"
        query = quote_plus(keyword)
        date_from = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        date_to = datetime.utcnow().strftime("%Y-%m-%d")
        params = {
            "q": query,
            "from": date_from,
            "to": date_to,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": page_size,
            "apiKey": os.environ["NEWS_API_KEY"],
        }
        response = requests.get(base_url, params=params)
        data = response.json()
        if data.get("status") != "ok":
            return []
        articles = data.get("articles", [])
        return [
            {
                "title": article.get("title"),
                "description": article.get("description"),
                "url": article.get("url"),
                "source": article.get("source", {}).get("name"),
                "publishedAt": article.get("publishedAt"),
                "content": article.get("content"),
            }
            for article in articles
        ]

    def select_relevant_articles(self, question: str, articles: list[dict]) -> list[dict]:
        headlines_list = [
            f"{i}: {article['title']} (Date: {article.get('publishedAt')})"
            for i, article in enumerate(articles)
        ]
        headlines_text = "\n".join(headlines_list)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a news expert helping to choose the most relevant headlines for a given question."),
            ("user", f"""
I have the following question and a list of news article headlines. Please tell me which headlines (by their indices) are most likely to help answer the question. Return only the indices as a comma-separated list.
Consider the publishing dates when making a decision. More recent articles are likely to be more useful, but it depends on the question.
Return about 5 selected articles.

Question: {question}

Headlines:
{headlines_text}
            """)
        ])
        messages = prompt.format_messages()
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0.7,
        )
        response = llm.invoke(messages, max_tokens=100).content
        indices = []
        for part in response.split(","):
            try:
                idx = int(part.strip())
                if 0 <= idx < len(articles):
                    indices.append(idx)
            except ValueError:
                continue
        selected = [articles[i] for i in indices]
        return selected

    def answer_question_with_news(self, question: str) -> tuple[float, float]:
        keywords = self.extract_news_keywords(question)
        all_articles = []
        for keyword in keywords:
            articles = self.fetch_articles_for_keyword(keyword)
            all_articles.extend(articles)
        
        selected_articles = self.select_relevant_articles(question, all_articles)
        
        contents = []
        for article in selected_articles:
            try:
                extraction = self.goose.extract(url=article["url"])
                cleaned_text = extraction.cleaned_text.strip()
                if cleaned_text:
                    contents.append(cleaned_text)
            except Exception as e:
                print(f"Article at {article['url']} could not be retrieved: {e}")
        concatenated_content = "\n\n".join(contents)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional prediction market trading agent."),
            ("user", f"""
Today is {utcnow()}.

Using the following news article contents, please answer the question:
Question: {question}

News Content:
{concatenated_content}

Return only the probability (as a float) and confidence (as a float), separated by a space.
            """)
        ])
        messages = prompt.format_messages()
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=1,
        )
        answer_response = llm.invoke(messages, max_tokens=256).content
        probability, confidence = map(float, answer_response.split())
        return probability, confidence

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        question = market.question
        probability, confidence = self.answer_question_with_news(question)
        return ProbabilisticAnswer(
            confidence=confidence,
            p_yes=Probability(probability),
            reasoning="I carefully reviewed relevant news articles and performed a detailed analysis to form this estimate.",
        )

if __name__ == "__main__":
    agent = YourAgent()
    agent.run(market_type=MarketType.OMEN)
