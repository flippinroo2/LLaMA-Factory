import sys
from enum import Enum, unique
from typing import Any, Dict, List, Optional

from transformers import TrainerCallback
from yaml import safe_load

# from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras.callbacks import LogCallback

# from llamafactory.extras.logging import get_logger
from llamafactory.hparams import get_train_args  # get_infer_args

# from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.dpo import run_dpo
from llamafactory.train.kto import run_kto
from llamafactory.train.orpo import run_orpo
from llamafactory.train.ppo import run_ppo
from llamafactory.train.pt import run_pt
from llamafactory.train.rm import run_rm
from llamafactory.train.sft import run_sft
from llamafactory.webui.interface import run_web_ui


@unique
class Command(str, Enum):
    EVAL = "eval"
    EXPORT = "export"
    TRAIN = "train"
    WEBDEMO = "webchat"
    WEBUI = "webui"
    HELP = "help"


DEFAULT_MODEL = "BLOOM-560M"


class TrainingWrapper:
    def __init__(self, args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:
        model_name: str = input("\nEnter model name: ")
        if model_name == "":
            print(f"\nNo model name entered. Using {DEFAULT_MODEL} as default.\n")
            train_configuration_file_name = "bloom_training_config.yaml"
        config = None
        with open(f"./config/{train_configuration_file_name}", "r", encoding="utf-8") as f:
            config = safe_load(f)

        model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(config)
        callbacks.append(LogCallback(training_args.output_dir))

        if finetuning_args.stage == "pt":
            TrainingWrapper.run_pretraining(model_args, data_args, training_args, finetuning_args, callbacks)
        elif finetuning_args.stage == "sft":
            TrainingWrapper.run_supervised_fine_tuning(
                model_args, data_args, training_args, finetuning_args, generating_args, callbacks
            )
        elif finetuning_args.stage == "rm":
            TrainingWrapper.run_reward_modeling(model_args, data_args, training_args, finetuning_args, callbacks)
        elif finetuning_args.stage == "ppo":
            TrainingWrapper.run_proximal_policy_optimization(
                model_args, data_args, training_args, finetuning_args, callbacks
            )
        elif finetuning_args.stage == "dpo":
            TrainingWrapper.run_distributed_policy_optimization(
                model_args, data_args, training_args, finetuning_args, callbacks
            )
        elif finetuning_args.stage == "kto":
            TrainingWrapper.run_knowledge_transfer_optimization(
                model_args, data_args, training_args, finetuning_args, callbacks
            )
        elif finetuning_args.stage == "orpo":
            TrainingWrapper.run_online_reward_prediction_optimization(
                model_args, data_args, training_args, finetuning_args, callbacks
            )
        else:
            raise ValueError("Unknown task.")

    @staticmethod
    def run_pretraining(model_args, data_args, training_args, finetuning_args, callbacks) -> None:
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)

    @staticmethod
    def run_supervised_fine_tuning(
        model_args, data_args, training_args, finetuning_args, generating_args, callbacks
    ) -> None:
        run_sft(model_args, data_args, training_args, finetuning_args, callbacks)

    @staticmethod
    def run_reward_modeling(model_args, data_args, training_args, finetuning_args, callbacks) -> None:
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)

    @staticmethod
    def run_proximal_policy_optimization(model_args, data_args, training_args, finetuning_args, callbacks) -> None:
        # PPO is an on-policy RL algorithm that optimizes a policy network to make decisions or take actions in an environment.
        # It is an extension of the Trust Region Policy Optimization (TRPO) algorithm, which aims to improve sample efficiency and training stability.
        # PPO uses a clipped surrogate objective function that encourages the new policy to stay close to the old policy, promoting stable and effective learning.
        # During finetuning, PPO can be used to train the LLM to generate more relevant and contextually appropriate responses or to optimize its behavior for a specific task.
        run_ppo(model_args, data_args, training_args, finetuning_args, callbacks)

    @staticmethod
    def run_distributed_policy_optimization(model_args, data_args, training_args, finetuning_args, callbacks) -> None:
        # DPO is a model-free, off-policy RL algorithm that optimizes a policy by directly modeling the distribution of returns (cumulative rewards) in an environment.
        # It extends the idea of distributional reinforcement learning, which focuses on estimating the entire distribution of returns rather than just the expected return.
        # DPO aims to capture the full range of possible outcomes and their probabilities, allowing for a more nuanced understanding of the environment.
        # During finetuning, DPO can help the LLM learn a more robust and generalized policy by considering the full spectrum of rewards and their probabilities.
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)

    @staticmethod
    def run_knowledge_transfer_optimization(model_args, data_args, training_args, finetuning_args, callbacks) -> None:
        # Knowledge Transfer:
        # Knowledge transfer involves leveraging the knowledge learned by one model (often a large pre-trained model) and transferring it to another model, usually a smaller or task-specific model.
        # The idea is to take advantage of the general language understanding and representations learned by large pre-trained LLMs and apply them to a new model designed for a specific task or domain.
        # Knowledge transfer can be achieved through various techniques, including pretraining and fine-tuning, distillation, or using pre-trained embeddings or representations as a starting point for the new model.
        # Optimization:
        # Optimization refers to the process of fine-tuning or training the model to optimize its performance on a specific task or dataset.
        # This involves updating the model's parameters based on a particular objective or loss function that aligns with the desired task.
        # Optimization techniques used in LLM finetuning can include gradient descent, backpropagation, or more advanced methods such as Adam, Adagrad, or RMSprop.
        # The goal of optimization is to adjust the model's weights or parameters in a way that minimizes the difference between predicted outputs and true outputs, improving the model's performance.

        # Here's a simplified overview of how KTO might be applied in LLM finetuning:

        # Start with a pre-trained LLM, such as GPT-3 or BERT, which has already learned a wealth of knowledge from large-scale text data.
        # Define the specific task or domain for which you want to fine-tune the model, such as sentiment analysis, question answering, or text classification.
        # Design a new model architecture or modify the existing LLM architecture to be suitable for the specific task. This might involve adding new layers, modifying existing layers, or changing the output structure.
        # Use knowledge transfer techniques to initialize the new model with the knowledge from the pre-trained LLM. This could include loading pre-trained weights, using pre-trained embeddings, or distilling knowledge from the teacher model to the student model.
        # Fine-tune the new model on a task-specific dataset using optimization techniques. Provide labeled examples for the specific task and update the model's parameters through backpropagation and gradient descent to minimize a defined loss function.
        # Evaluate the model's performance on a validation set and iteratively adjust hyperparameters, model architecture, or training data as needed to optimize performance.
        # Deploy the fine-tuned model to perform the specific task, leveraging the transferred knowledge and optimized parameters.
        run_kto(model_args, data_args, training_args, finetuning_args, callbacks)

    @staticmethod
    def run_online_reward_prediction_optimization(
        model_args, data_args, training_args, finetuning_args, callbacks
    ) -> None:
        run_orpo(model_args, data_args, training_args, finetuning_args, callbacks)


def webui_function() -> None:
    run_web_ui()


def main() -> None:
    program_arguments: list[str] = sys.argv
    valid_command: bool = False  # Default for "valid_command" so the loop runs at least one time.
    while not valid_command:
        valid_command = True  # Setting command to valid unless it's actually not valid.
        command: str | None = None
        if len(program_arguments) > 1:
            command = program_arguments.pop(1).lower()
        else:
            command = input("Enter command to run: ").lower()

        if command == Command.TRAIN:
            TrainingWrapper()
        elif command == Command.WEBUI:
            webui_function()
        # elif command == Command.EVAL:
        #     run_eval()
        # elif command == Command.EXPORT:
        #     export_model()
        else:
            print("Defaulting to TRAIN command.\n")
            TrainingWrapper()
            # print(f'\nUnknown command: "{command}"\n')
            # valid_command = False  # Setting command to not valid again.


if __name__ == "__main__":
    main()
