from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.utils import utcnow
from prediction_market_agent_tooling.deploy.betting_strategy import (
    BettingStrategy,
    KellyBettingStrategy,
)


from openai import OpenAI

class OpenaiSearchAgentVariable(DeployableTraderAgent):
    bet_on_n_markets_per_run = 1

    # # loader for variables, compliant with interface
    # def load(self, reasoning: str = "low", search_context_size: str = "low"):
    #     self.reasoning = reasoning
    #     self.search_context_size = search_context_size
    #     self.custom_agent_name = f"{self.__class__.__name__} (reasoning={reasoning}, search_context_size={search_context_size})"

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        """Return a simple Kelly strategy with a fixed bet cap.

        We drop the dynamic balance lookup to avoid the heavy dependency on
        prediction_market_agent.* utilities that require a Postgres-backed
        memory layer.  Adjust MAX_BET if you want the agent to be more or
        less aggressive.
        """
        MAX_BET = 0.01  # xDai
        return KellyBettingStrategy(max_bet_amount=MAX_BET, max_price_impact=0.7)

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        client = OpenAI()
        today=utcnow()

        # Ask search API for a report
        search_response = client.responses.create(
            model="gpt-4o",
            tools=[{
                "type": "web_search_preview",
                "search_context_size": "high",
            }],
            input=[
                {
                    "role": "developer",
                    "content": f"""Today is {today}.

                You will be given a question in the following user message. Your task is to extract, compile, and organize every piece of relevant information that could help a reasoning model assess the likelihood of the described event occurring. The final outcome should include:

                - A comprehensive set of evidence without omitting any pertinent details (unless they are obviously irrelevant). Do not provide links, explain the evidence in it's entirety.
                - No conclusions or judgments unless the evidence overwhelmingly points to one.
                - A clear presentation of all evidence, which will later be used to derive a probability and confidence level for the event.

                Focus exclusively on presenting the evidence and avoid speculative analysis."""
                },
                {
                    "role": "user",
                    "content": f"{market.question}"
                }
            ]
        )

        # comprehensize report from search LLM
        context = search_response.output_text

        # ask reasoning model for a probability estimate given context from report
        reasoning_response = client.responses.create(
            model="o3-mini",
            input=[
                {
                    "role": "developer",
                    "content": f"""Today is {today}.

    Given the following question and information from the web, what's the probability that the thing in the question will happen?.
    
    Return only the probability float number and confidence float number, separated by space, nothing else."""
                },
                {
                    "role": "user",
                    "content": f"""Question: {market.question}

    Context: {context}"""
                }
            ],
            reasoning={"effort": "high"}
        )

        # eval_text = (
        #     f"{market.question}\n"
        #     f"{search_response.output_text}\n"
        #     f"{reasoning_response.output_text}\n"
        #     f"Reasoning: {self.reasoning} Search context: {self.search_context_size}\n"
        #     f"Search input: {search_response.usage.input_tokens}\n"
        #     f"Search output: {search_response.usage.output_tokens}\n"
        #     f"Reasoning input: {reasoning_response.usage.input_tokens}\n"
        #     f"Reasoning output: {reasoning_response.usage.output_tokens}\n"
        #     f"Total tokens: {search_response.usage.total_tokens + reasoning_response.usage.total_tokens}\n"
        # )

        # with open("evals.txt", "a") as file:
        #     file.write(eval_text)

        probability_and_confidence = str(reasoning_response.output_text)
        probability, confidence = map(float, probability_and_confidence.split())

        return ProbabilisticAnswer(
            confidence=confidence,
            p_yes=Probability(probability),
            reasoning="I asked Google and LLM to do it!",
        )


if __name__ == "__main__":
    agent = OpenaiSearchAgentVariable()
    agent.run(market_type=MarketType.OMEN)
