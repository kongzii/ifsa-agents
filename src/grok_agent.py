from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.gtypes import Probability




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
    
