import argparse
from openai import OpenAI

import os
import numpy as np
import pandas as pd
import time
from urllib import request
import tarfile

from utils import crop

# Or use your own model
client = OpenAI(api_key="INSERTYOURKEYHERE")
choices = ["A", "B", "C", "D"]


# Download and extract MMLU Dataset
def download_and_extract_mmlu(dir):
    mmlu = os.path.join(dir, "data.tar")
    request.urlretrieve(
        "https://people.eecs.berkeley.edu/~hendrycks/data.tar", mmlu
    )
    mmlu_tar = tarfile.open(mmlu)
    mmlu_tar.extractall(dir)
    mmlu_tar.close()


# Helper functions
def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    return softmax


def format_subject(subject):
    entries = subject.split("_")
    s = ""
    for entry in entries:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = (
        "The following are multiple choice questions "
        "(with answers) about {}.\n\n".format(format_subject(subject))
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


# Here we will evaluate the model against the dataset
def eval(args, subject, engine, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # Generate prompt from dataset
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1] - 1]

        # Prompt your model
        while True:
            try:
                c = client.completions.create(
                    model=engine,
                    prompt=prompt,
                    max_tokens=1,
                    logprobs=100,
                    temperature=0,
                    echo=True,
                )  # Use your model here!
                break
            except Exception:
                print("pausing")
                time.sleep(1)
                continue

        # Evaluate models response to questions answer
        lprobs = []
        for ans in answers:
            try:
                lprobs.append(
                    c.choices[0].logprobs.top_logprobs[-1][
                        " {}".format(ans)
                    ]
                )
            except Exception:
                print(
                    "Warning: {} not found. "
                    "Artificially adding log prob of -100.".format(ans)
                )
                lprobs.append(-100)
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))

        # Record results
        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    # Summarize results
    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


# Run Evaluations
def main(args):
    engines = args.engine

    # Download and prepare dataset subjects
    download_and_extract_mmlu(args.data_dir)
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "data/test"))
            if "_test.csv" in f
        ]
    )

    # Create directories to save results to if they don't exist
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    for engine in engines:
        if not os.path.exists(
            os.path.join(args.save_dir, "results_{}".format(engine))
        ):
            os.mkdir(
                os.path.join(args.save_dir, "results_{}".format(engine))
            )

    print(subjects)
    print(args)

    # Run evaluations for each engine
    for engine in engines:
        print(engine)
        all_cors = []

        for subject in subjects:
            # Prepare examples for prompting
            dev_df = pd.read_csv(
                os.path.join(
                    args.data_dir, "data/dev", subject + "_dev.csv"
                ),
                header=None,
            )[: args.ntrain]
            # Prepare actual test questions
            test_df = pd.read_csv(
                os.path.join(
                    args.data_dir, "data/test", subject + "_test.csv"
                ),
                header=None,
            )

            # Run evaluations
            cors, acc, probs = eval(args, subject, engine, dev_df, test_df)
            all_cors.append(cors)

            # Save results
            test_df["{}_correct".format(engine)] = cors
            for j in range(probs.shape[1]):
                choice = choices[j]
                test_df["{}_choice{}_probs".format(engine, choice)] = probs[
                    :, j
                ]
            test_df.to_csv(
                os.path.join(
                    args.save_dir,
                    "results_{}".format(engine),
                    "{}.csv".format(subject),
                ),
                index=None,
            )

        weighted_acc = np.mean(np.concatenate(all_cors))
        print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ntrain",
        "-k",
        type=int,
        default=5,
        help="Number of samples to include for n-shot prompting",
    )
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument(
        "--engine",
        "-e",
        choices=["davinci", "curie", "babbage", "ada"],
        default=["davinci", "curie", "babbage", "ada"],
        nargs="+",
        help="OpenAI model(s) you would like to evaluate",
    )
    args = parser.parse_args()
    main(args)
