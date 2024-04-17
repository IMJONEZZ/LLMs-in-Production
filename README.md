# LLMs In Production

A repository for the book _LLMs in Production_ with Manning Publications

Chapter listings will be kept here.

Please consider purchasing the book [here](http://mng.bz/OPGP):

<img src="https://api.qrserver.com/v1/create-qr-code/?size=200x200&bgcolor=FFFFFF&data=https%3A%2F%2Fwww.manning.com%2Fbooks%2Fllms-in-production%3Futm_source%3Dmsharp9%26utm_medium%3Daffiliate%26utm_campaign%3Dbook_brousseau_llms_9_27_23%26a_aid%3Dmsharp9%26a_bid%3Dba4fb1b2" />


## How to use this repo

Create an environment and install dependencies:
```bash
make setup
```
This will create an environment for each chapter, named after the chapter directory, e.g. chapter_1.

Activate environment:
```bash
conda activate llmbook
```

Deactivate environment:
```bash
conda deactivate
```

Run linters and formatters:
```bash
make lint
```

Run Tests:
```bash
make test
```

Remove all environments:
```bash
make clean
```


## Additional notes

If necessary, each chapter will contain its own README.md file with additional setup instructions.

Some listings are boilerplates and are not intended to be ran. When possible, examples are given that can be ran for additional context.

All scripts are designed to be ran from project root, e.g. `python chapters/chapter_1/listing_1.1.py`



## Manning Publications

Check out other Manning titles and learning resources [here](https://tinyurl.com/5x2h9k4y)!