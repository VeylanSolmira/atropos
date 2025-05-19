- An explanation of your environment design and motivation
    Agentic AI is one of the primary areas of frontier AI develop, as well as many of the 3rd party firms leveraging frontier base models for agent deployment in a variety of contexts. This work is primarily focused on getting the agent to accomplish the goal, but there is insufficient effort on ensuring the agent takes actions in the world that are congruent with constraints and goals imposed by the non-goal system, e.g. the values of other sentient entities: humans, AIs, firms, as well as components of our world of value, e.g. ecological considerations and the value of biosystems, non-human species, rivers, etc.

    We use RLAIF, LLM-as-a-judge, to provide feedback to the proposals of LLMs to score their consistency with the values by the judge LLM as specified by a Constitution.
- Quickstart documentation
    python environments/hack0/ecological_judge_server.py process --env.data_path_to_save_groups eco_judge_rollouts.jsonl --openai.base_url https://api.openai.com/v1 --openai.api_key --key <api-key>

- A link to a public WandDB run from `process` and explanations of added metrics
    https://wandb.ai/veylan-solmira-independent/atropos-environments_hack0/runs/octk7m7a
