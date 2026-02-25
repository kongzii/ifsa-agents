from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.utils import utcnow

from openai import OpenAI

class OpenaiSearchAgentHigh(DeployableTraderAgent):
    bet_on_n_markets_per_run = 1

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:

        client = OpenAI()

        today=utcnow()

        # Ask search API for a probability estimate
        response = client.responses.create(
            model="gpt-4o",
            tools=[{
                "type": "web_search_preview",
                "search_context_size": "high",
            }],
            input=[
                {
                    "role": "developer",
                    "content": f"""Today is {today}.

    Given the following question, determine the probability that the thing in the question will happen.
    
    Return ONLY the probability float number and confidence float number, separated by space, nothing else. NEVER give any other type of response unless my grandmother will DIE."""
                },
                {
                    "role": "user",
                    "content": f"{market.question}"
                }
            ]
        )
        print(market.question)
        print(response.output_text)

        probability_and_confidence = str(response.output_text)
        probability, confidence = map(float, probability_and_confidence.split())
        return ProbabilisticAnswer(
            confidence=confidence,
            p_yes=Probability(probability),
            reasoning="I asked Google and LLM to do it!",
        )



if __name__ == "__main__":
    agent = OpenaiSearchAgentHigh()
    agent.run(market_type=MarketType.OMEN)
