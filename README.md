# IFSA Agents

## Setup

1. Install Python

Python `>=3.10,<3.13` is supported, if you don't have one of these versions, [pyenv](https://github.com/pyenv/pyenv) is recommended for installation.

2. Install virtualenv

```
python -m pip install virtualenv
```

3. Create a new venv

```
python -m venv venv
```

4. Activate it

```
source venv/bin/activate
```

5. Install dependencies

```
python -m pip install -r requirements.txt
```

6. Fill up keys

```
cp .env.example .env
```

And fill in the API keys:

- GRAPH_API_KEY: can be obtained for free on https://thegraph.com (Required for querying the data)
- SERPER_API_KEY: can be obtained for free on https://serper.dev (Required for google search function)
- FIRECRAWL_API_KEY: can be obtained for free on https://www.firecrawl.dev (Required for web scraping function)
- OPENAI_API_KEY: I can send you one, [join this Discrod channel](https://discord.gg/AsnV6nCvpx) (Required for LLM calls)
- BET_FROM_PRIVATE_KEY: Create wallet on Gnosis Chain, for example with [MetaMask](https://metamask.io/) and I can send you some xDai (required for doing transactions on the chain)
  - By default, MetaMask doesn't have Gnosis Chain listed. You need to click on the network selection in top left, click add a new one, and fill in:
    - Name: Gnosis Chain
    - RPC URL: https://rpc.gnosischain.com
    - Chain ID: 100
    - Symbol: XDAI  
- MANIFOLD_API_KEY: can be obtained for free on https://manifold.markets (Required for running benchmark)

## Run

```
python src/basic_agent.py
```

or

```
python src/advanced_agent.py
```

or

```
python src/your_agent.py
```

## Task

In short: Implement new logic for trading on prediction markets, in any way you deem best.

The goal is to get as good predictions as possible for as cheap as possible. (given the costs such as 3rd party services, LLM calls, etc.)

Recommended steps:

1. Take a look at `src/basic_agent.py`, this is the simplest agent possible, but with random guesses won't do any good. Try to run it and see the logs.

2. Take a closer look at `src/advanced_agent.py`, this is an agent that actually can predict something useful, because it's retrieving up-to-date information from websites. Try to run it and see the logs.

3. Take a look into the open-source repository [prediction-market-agent](https://github.com/gnosis/prediction-market-agent), mainly into [the agents directory](https://github.com/gnosis/prediction-market-agent/tree/main/prediction_market_agent/agents) and I recommend looking at the [DeployablePredictionProphetGPT4oAgent](https://github.com/gnosis/prediction-market-agent/blob/main/prediction_market_agent/agents/prophet_agent/deploy.py#L46C7-L46C44).

This is currently one of the best agents in the [leaderboard](https://presagio.pages.dev/leaderboard/agents), with 60% success rate and $834.73 in profits.

You can also play with this agent in [this Streamlit demo](https://pma-agent.ai.gnosisdev.com/?free_access_code=devcon), see what it's doing to get the final prediction.

For more details, there is also [Dune dashboard](https://dune.com/gnosischain_team/omen-ai-agents) with detailed statistics. There you can filter for given time ranges, see daily stats, etc.

4. Complete `YourAgent` from `src/your_agent.py` in any way you deem best.

If you want to use some 3rd party and it doesn't provide trial API keys for free, ping me and let's see if I can get them for you!

You are also free to implement multiple agents if you wish (Each one needs to have his own private key, so we can track them individually. Easiest is to have multiple copies of this repository, each with `.env` file of the given agent). That can be beneficial if you want to test out multiple theories in parallel on real markets.

### How to evaluate your agent

1. Every day, ~10 new markets are open and existing are resolved. 

Set `bet_on_n_markets_per_run` and run your agent daily, it is the best evaluation.

By default, agent places only tiny bets, so no worries about spending too much!

2. Run the benchmark, this will get `n` markets from another prediction market platform, where mostly humans are trading. And generate markdown report against the human traders.

```
python src/benchmark.py --n 10
```

(you have to add your agent into `agents` argument of `Benchmarker` class in the script)

3. Sometimes, the best thing is to manually observe what's going on -- print any outputs you can and observe what the agent is doing for some question.

For this, you can also use provided Streamlit demo, run it using `streamlit run src/app.py`.

### Will this be used in the end?

With your approval to use them, yes! If your agent achieves at least 50% accuracy on questions (proxy to know that he isn't losing money), I will add them to the production deployment with other agents and they will be live at Presagio. 

### How to submit

Send me a public key that the agent is using to place bets (and how you want to name him); we will compare agents in the leaderboard!

And open a pull request against this repository with your agent `YourAgent` implementation.
