from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dsp.modules.lm import LM
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config,
    attn_implementation="sdpa",
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True,
)

gms8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gms8k.train[:30], gms8k.dev[:100]


def openai_to_hf(**kwargs):
    hf_kwargs = {}
    for k, v in kwargs.items():
        if k == "n":
            hf_kwargs["num_return_sequences"] = v
        elif k == "frequency_penalty":
            hf_kwargs["repetition_penalty"] = 1.0 - v
        elif k == "presence_penalty":
            hf_kwargs["diversity_penalty"] = v
        elif k == "max_tokens":
            hf_kwargs["max_new_tokens"] = v
        elif k == "model":
            pass
        else:
            hf_kwargs[k] = v

    return hf_kwargs


class HFModel(LM):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        **kwargs
    ):
        """wrapper for Hugging Face models

        Args:
            model(AutoModelForCausalLM): HF model identifier to load and use
            tokenizer: AutoTokenizer
        """
        super().__init__(model)
        self.model = model
        self.tokenizer = tokenizer
        self.drop_prompt_from_output = True
        self.history = []
        self.is_client = False
        self.device = model.device
        self.kwargs = {
            "temperature": 0.3,
            "max_new_tokens": 300,
        }

    def basic_request(self, prompt, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        response = self._generate(prompt, **kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    def _generate(self, prompt, **kwargs):
        kwargs = {**openai_to_hf(**self.kwargs), **openai_to_hf(**kwargs)}
        if isinstance(prompt, dict):
            try:
                prompt = prompt["messages"][0]["content"]
            except (KeyError, IndexError, TypeError):
                print("Failed to extract 'content' from the prompt.")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # print(kwargs)
        outputs = self.model.generate(**inputs, **kwargs)
        if self.drop_prompt_from_output:
            input_length = inputs.input_ids.shape[1]
            outputs = outputs[:, input_length:]
        completions = [
            {"text": c}
            for c in self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
        ]
        response = {
            "prompt": prompt,
            "choices": completions,
        }
        return response

    def __call__(
        self, prompt, only_completed=True, return_sorted=False, **kwargs
    ):
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        if kwargs.get("n", 1) > 1 or kwargs.get("temperature", 0.0) > 0.1:
            kwargs["do_sample"] = True

        response = self.request(prompt, **kwargs)
        return [c["text"] for c in response["choices"]]


# Set up the LM
print("Model set up!")
llama = HFModel(model, tokenizer)

# Set up DSPY to use that LM
dspy.settings.configure(lm=llama)


# Define the QASignature and CoT (Chain-of-Thought)
class QASignature(dspy.Signature):
    (
        """You are given a question and answer"""
        """and you must think step by step to answer the question. """
        """Only include the answer as the output."""
    )
    question = dspy.InputField(desc="A math question")
    answer = dspy.OutputField(desc="An answer that is a number")


class ZeroShot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(QASignature, max_tokens=1000)

    def forward(self, question):
        return self.prog(question=question)


# Set up the evaluator, which can be used multiple times.
evaluate = Evaluate(
    devset=gsm8k_devset,
    metric=gsm8k_metric,
    num_threads=4,
    display_progress=True,
    display_table=0,
)

# Evaluate how the LLM does with no changes
print("Evaluating Zero Shot")
evaluate(ZeroShot())  # 29/200 14.5%

# Set up the optimizer
config = dict(max_bootstrapped_demos=2)


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(QASignature, max_tokens=1000)

    def forward(self, question):
        return self.prog(question=question)


# Optimize the prompts
print("Creating Bootstrapped Few Shot Prompt")
teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
optimized_cot = teleprompter.compile(
    CoT(), trainset=gsm8k_trainset, valset=gsm8k_devset
)
optimized_cot.save("optimized_llama3_math_cot.json")

# Evaluate our `optimized_cot` program.
print("Evaluating Optimized CoT Prompt")
evaluate(optimized_cot)  # 149/200 74.5%
