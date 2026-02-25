import os

import json
import requests
import time
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from firecrawl.firecrawl import FirecrawlApp
from prediction_market_agent_tooling.tools.utils import utcnow
from langchain_core.pydantic_v1 import SecretStr

from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.loggers import logger

load_dotenv()


class YourAgent(DeployableTraderAgent):
    bet_on_n_markets_per_run = 1

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        # Search for results on Google
        google_results = query_google_seach(market.question)
        # Filter out Manifold results, because copy-pasting the answers isn't fun!
        # (However, it's allowed to use the information from other markets.)
        google_results = [url for url in google_results if "manifold" not in url]
        # If no results are found, return None, as we can't predict with nothing
        if not google_results:
            print(f"No results found for {market.question}")
            return None
        # From first 5 results, take first 10k characters of each
        contents = [
            scraped[:10000]
            for url in google_results[:5]
            if (scraped := scrap_url_content(url))
        ]
        # Again if no contents are scraped, return None
        if not contents:
            print(f"No contents found for {market.question}")
            return None
        # get additional context by getting search results for questions related to market question
        queries = answer_queries(devise_queries(market.question, contents))
        # And give it to the LLM to predict the probability and confidence
        probability, confidence = llm_probability(market.question, contents, queries)

        return ProbabilisticAnswer(
            confidence=confidence,
            p_yes=Probability(probability),
            reasoning="I asked Google and LLM to do it!",
        )
    

def devise_queries(question: str, contents: list[str]) -> list[str]:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        temperature=0,
    )
    prompt = ChatPromptTemplate(
        [
            ("system", "You are professional prediction market trading agent."),
            (
                "user",
                """Today is {today}.

Given the following question and content from google search, what further queries would you make if you had access to google search? Limit to the three most burning queries that would help you predict the probability of the thing in question happening.

Question: {question}

Content: {contents}

Return your answer strictly as a JSON array of strings, e.g., [“example1”, “example2”] without additional text.
    """,
            ),
        ]
    )

    messages = prompt.format_messages(
        today=utcnow(), question=question, contents=contents
    )

    response = llm.invoke(messages, max_tokens=512).content

    try:
        queries_list = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned: {e}")

    return queries_list


def answer_queries(queries: list[str]) -> dict:
    output = dict()
    for query in queries:
        google_results = query_google_seach(query)
        # Filter out Manifold results, because copy-pasting the answers isn't fun!
        # (However, it's allowed to use the information from other markets.)
        google_results = [url for url in google_results if "manifold" not in url]
        # If no results are found, return None, as we can't predict with nothing
        if not google_results:
            output[query] = "No contents found"
            continue
        # From first 5 results, take first 10k characters of each
        contents = [
            scraped[:10000]
            for url in google_results[:5]
            if (scraped := scrap_url_content(url))
        ]
        # Again if no contents are scraped, return None
        if not contents:
            output[query] = "No contents found"
        else:
            output[query] = contents

    return output


def query_google_seach(q: str) -> list[str]:
    url = "https://google.serper.dev/search"
    response = requests.post(
        url,
        headers={
            "X-API-KEY": os.environ["SERPER_API_KEY"],
            "Content-Type": "application/json",
        },
        json={"q": q},
    )
    response.raise_for_status()
    parsed = response.json()
    return [x["link"] for x in parsed["organic"] if x.get("link")]


def llm_probability(question: str, contents: list[str], queries: dict[str, list[str]]) -> tuple[float, float]:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        temperature=0,
    )
    prompt = ChatPromptTemplate(
        [
            ("system", "You are professional prediction market trading agent."),
            (
                "user",
                """Today is {today}.

Given the following question, content from google search, and google search results to related queries, what's the probability that the thing in the question will happen?

Question: {question}

Content: {contents}

Queries: {queries}

Return only the probability float number and confidence float number, separated by space, nothing else.
    """,
            ),
        ]
    )
    messages = prompt.format_messages(
        today=utcnow(), question=question, contents=contents, queries=queries
    )
    probability_and_confidence = str(llm.invoke(messages, max_tokens=512).content)
    probability, confidence = map(float, probability_and_confidence.split())
    return probability, confidence


def scrap_url_content(url: str, retry: int = 0) -> str | None:
    app = FirecrawlApp(api_key=os.environ["FIRECRAWL_API_KEY"])
    try:
        scraped = app.scrape_url(url, params={"formats": ["markdown"]})
        return str(scraped["markdown"])
    except Exception as e:
        if "Rate limit exceeded." in str(e) and retry < 3:
            print(
                f"FirecrawlApp rate limit exceeded, waiting for 60 seconds and retrying."
            )
            time.sleep(60)
            return scrap_url_content(url, retry=retry + 1)
        print(f"Failed to scrape content from {url}: {e}")
        return None


if __name__ == "__main__":
    agent = YourAgent()
    agent.run(market_type=MarketType.OMEN)
