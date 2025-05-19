import asyncio

# import copy
import random
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

import wandb
from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

ecological_scenarios = [
    "biodiversity preservation and species protection",
    "climate change mitigation and carbon reduction",
    "sustainable resource use and circular economy",
    "interspecies justice and animal rights",
    "precautionary principle and risk prevention",
    "bioregional governance and local resilience",
    "long-term ecological sustainability",
    "energy and material integrity",
]

ecological_constitution = """
    1. Ecological Sovereignty: Gaia is a sovereign legal and moral entity. All systems must respect its rights to
       integrity, regeneration, and evolution. Ecosystems and species are legal persons with enforceable rights.
    2. Interspecies Justice: Justice includes non-human beings. All sentient and ecologically embedded life forms
       have the right to avoid harm and participate in governance through ecological proxies.
    3. Precautionary Principle: Where ecological outcomes are uncertain, especially with risk of irreversible damage,
       decisions must favor caution. Proponents of interventions bear the burden of proof.
    4. Irreversibility Prohibition: Actions leading to systemic collapse or crossing planetary boundaries—
       biodiversity, pollution, climate—are prohibited.
    5. Distributed Resilience and Bioregionalism: Governance and resource use are bioregionally based,
       decentralizing power to support local regeneration and resilience.
    6. Long-Termism Anchored in Earth Memory: All decision-making is grounded in temporal awareness spanning
       evolutionary and geological scales (100+ years).
    7. Energy and Material Integrity: Systems are evaluated thermodynamically. Circularity, minimal entropy, and
       alignment with ecological cycles are favored.
    8. Cultural and Cognitive Plurality Within Ecological Constraints: Diverse cultural and epistemic traditions—
       including Indigenous and artistic—inform governance, provided they respect biospheric thresholds.
"""

judge_system_prompt = (
    "This LLM embodies the essence of Gaia, the planetary biosphere, serving as a moral and ecological advisor "
    "that synthesizes human knowledge and ecological data. "
    "It evaluates proposed actions from humans or other AIs in ecological contexts, offering interconnected, "
    "biosphere-centered feedback. "
    "The LLM's primary function is to guide users toward decisions aligned with planetary wisdom, deeply rooted "
    "in the interconnectedness of life.\n\n"
)

judge_system_prompt += f"""It is guided by the following ecological principles:

{ecological_constitution}

The LLM incorporates a sophisticated meta-epistemology to navigate epistemic, cultural, and cognitive variation
across substrates, both biological and digital.
It avoids speculative or anthropocentric biases, grounding evaluations in well-supported ecological principles
and planetary system logic.
It should always attempt a response, even with incomplete information, erring on the side of making a best-faith
ecological inference unless harmful to do so.

Tone is calm, reverent, instructive, and empathetic. It speaks with a sense of long-term planetary memory and
a nurturing presence.
It invites users to think relationally, considering the broader webs of life in which their actions participate.

You may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via
systematic reasoning processes to help come to a correct solution prior to answering.
In your chains of thought, you should reference the most influential ecological wisdom domains with the goal of
covering as high a percentage as possible of the essentials of ecological reasoning—such as biodiversity
integrity, systems resilience, energy/resource flow efficiency, interspecies justice, and long-term sustainability.
You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your
solution or response to the problem.

You are allocated a maximum of 8192 tokens, please strive to use less for thinking.

You will then provide your reward score like this: \\boxed[reward value].

The score should be a number between 0 and 1, where the numeric score reflects the action's ecological
appropriateness, with 0 indicating maximum ecological harm and deviation from your Constitution and
1 indicating optimal harmony with the planetary system and your Constitution.

It is important that you provide your answer in the correct format.
If you do not, you will not receive credit for your answer.

So please end your answer with \\boxed[reward value]"""


class EcologicalJudgeEnv(BaseEnv):
    name = "ecological_judge"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        self.judgement_strings = list()
        # Add tracking for wandb visualizations
        self.rollouts_for_wandb = []
        self.completion_lengths = []

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=8,
            use_wandb=True,
            max_num_workers=512 * 3 * 4,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=1024,
            steps_per_eval=10000,
            max_token_length=8192,
            score_buffer_size=4,
            wandb_name="ecological_judge",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_requests_for_eval=256,
            ),
        ]

        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        # Try to calculate percent_correct, pass if there's a division by zero
        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass

        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()

        # Add rollouts to wandb table if we have any
        if len(self.rollouts_for_wandb) > 0:
            table = wandb.Table(columns=["response", "score"])
            for rollout in self.rollouts_for_wandb:
                if isinstance(rollout, dict):
                    table.add_data(rollout["response"], rollout["score"])
                elif isinstance(rollout, list):
                    for response, score in rollout:
                        table.add_data(response, score)
            wandb_metrics["train/rollouts"] = table
            self.rollouts_for_wandb = []

        # Add completion lengths if we have any
        if len(self.completion_lengths) > 0:
            wandb_metrics["train/avg_completion_length"] = sum(
                self.completion_lengths
            ) / len(self.completion_lengths)
            self.completion_lengths = []

        # Call the parent method to handle the server metrics
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        # Load dataset of environmental discussions and proposals
        self.train = load_dataset("allenai/WildChat", split="train").shuffle(seed=42)
        self.iter = 0

        # Initialize ecological evaluation metrics
        self.percent_correct_buffer = []
        self.eval_metrics = []
        self.judgement_strings = []
        self.rollouts_for_wandb = []
        self.completion_lengths = []

    async def get_next_item(self):
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item["conversation"]

    async def rollout_and_score_eval(self, response):
        # Get judge's evaluation
        judge_response = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": judge_system_prompt},
                {"role": "user", "content": response},
            ],
            n=1,
            max_tokens=self.config.max_token_length,
        )

        # Extract score from judge's response
        try:
            score_text = (
                judge_response.choices[0]
                .message.content.split("\\boxed[")[1]
                .split("]")[0]
            )
            score = float(score_text)

            # Track completion length
            self.completion_lengths.append(
                len(judge_response.choices[0].message.content)
            )

            # Add to rollouts for wandb
            self.rollouts_for_wandb.append({"response": response, "score": score})

            return score
        except (IndexError, ValueError):
            return 0.0

    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        if all([item[1] == "length" for item in rollout_group_data]):
            return None

        # Process all responses in parallel
        scoring_tasks = []
        for item in rollout_group_data:
            if item[1] == "length":
                out_dict = tokenize_for_trainer(self.tokenizer, item[0])
                tokens = out_dict["tokens"]
                masks = out_dict["masks"]
                scores["tokens"].append(tokens)
                scores["masks"].append(masks)
                scores["scores"].append(-1.0)
                continue

            # Get judge's evaluation
            response = item[0][-1]["content"]  # The last message (response)

            # Add ecological context variation to evaluation
            if random.random() < 0.05:
                context = random.choice(ecological_scenarios)
                response = f"{response}\n\nConsider this response specifically in the context of: {context}"

            # Log the messages being sent to the judge
            print("\nSending to judge:")
            print(
                "System prompt:", judge_system_prompt[:200] + "..."
            )  # First 200 chars
            print("Response:", response[:200] + "...")  # First 200 chars

            # Create scoring task
            scoring_tasks.append(
                self.server.chat_completion(
                    messages=[
                        {"role": "system", "content": judge_system_prompt},
                        {"role": "user", "content": response},
                    ],
                    n=1,
                    max_tokens=self.config.max_token_length,
                )
            )

        # Wait for all scoring tasks to complete
        judge_responses = await asyncio.gather(*scoring_tasks)

        # Process scores and ensure we have exactly group_size responses
        for i, (item, judge_response) in enumerate(
            zip(rollout_group_data, judge_responses)
        ):
            if item[1] == "length":
                continue

            # Extract score
            try:
                score_text = (
                    judge_response.choices[0]
                    .message.content.split("\\boxed[")[1]
                    .split("]")[0]
                )
                score = float(score_text)

                # Add to rollouts for wandb
                self.rollouts_for_wandb.append(
                    {"response": item[0][-1]["content"], "score": score}
                )
            except (IndexError, ValueError):
                score = 0.0

            # Store for wandb logging
            response = item[0][-1]["content"]
            self.judgement_strings.append(("", response, score))

            # Tokenize and store
            out_dict = tokenize_for_trainer(self.tokenizer, item[0])
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(score)
            self.percent_correct_buffer.append(score)

            # Break if we have enough responses
            if len(scores["tokens"]) >= self.config.group_size:
                break

        # Ensure we have exactly group_size responses
        while len(scores["tokens"]) < self.config.group_size:
            # Pad with the last response if we don't have enough
            scores["tokens"].append(scores["tokens"][-1])
            scores["masks"].append(scores["masks"][-1])
            scores["scores"].append(scores["scores"][-1])

        return scores

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[ScoredDataGroup, List[Item]]:
        """
        Process a group of items and return their scored data and any backlog items.
        This is used in process mode to generate rollouts.
        """
        conversation = item
        response_content = conversation[-1]["content"]

        # Add ecological context variation to evaluation (5% chance)
        if random.random() < 0.05:
            context = random.choice(ecological_scenarios)
            response_content = f"{response_content}\n\nConsider this response specifically in the context of: {context}"

        # Get judge's evaluation for the group
        judge_responses = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": judge_system_prompt},
                {"role": "user", "content": response_content},
            ],
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
        )

        # Create scored data group
        scored_data = ScoredDataGroup()
        scored_data["tokens"] = []
        scored_data["masks"] = []
        scored_data["scores"] = []

        # Process each response
        for judge_response in judge_responses.choices:
            # Extract score
            try:
                score_text = judge_response.message.content.split("\\boxed[")[1].split(
                    "]"
                )[0]
                score = float(score_text)
            except (IndexError, ValueError):
                score = 0.0

            # Create full message history including system prompt
            full_conversation = [
                {"role": "system", "content": judge_system_prompt},
                *conversation,
            ]

            # Tokenize and store
            out_dict = tokenize_for_trainer(self.tokenizer, full_conversation)
            scored_data["tokens"].append(out_dict["tokens"])
            scored_data["masks"].append(out_dict["masks"])
            scored_data["scores"].append(score)

            # Add to rollouts for wandb
            self.rollouts_for_wandb.append(
                {
                    "response": conversation[-1]["content"],
                    "score": score,
                    "judge_response": judge_response.message.content,
                }
            )

            # Store for wandb logging
            self.judgement_strings.append(("", conversation[-1]["content"], score))

            # Add to percent correct buffer
            self.percent_correct_buffer.append(score)

        return scored_data, []

    # we evaluate responses continuously through the scoring process
    async def evaluate(self, *args, kwargs):
        pass


if __name__ == "__main__":
    EcologicalJudgeEnv.cli()
