# Unsloth Experience

The day came that I finally wanted to try the ultimate model trainer. The most memory efficient, 2x faster, 2.2x faster, 73% less, free notebooks -- Unsloth Trainer. 

The early sign that should have caught my attention that something is wrong is that they have just SO. MANY. NOTEBOOKS.

But I was optimistic and careless as I always am when trying a new tool. I mean it wasn't me who produced it, they surely knew what they were doing right?

## Unsloth ~~Trainer~~ Optimizer
So basically what Unsloth is, after interacting with it for an hour and with my 2head iq, is a Huggingface trainer wrapper with a custom optimizer. 

```python
# /opt/conda/lib/python3.10/site-packages/unsloth/trainer.py
class UnslothTrainer(SFTTrainer):
    def create_optimizer(self):
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None: return super().create_optimizer()

        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = _create_unsloth_optimizer(
                self.model,
                optimizer_cls,
                optimizer_kwargs,
                embedding_learning_rate,
            )
        pass
        return self.optimizer
    pass
pass
```

And also the `UnslothTrainingArguments`, are just Huggingface's `TrainingArguments` with a smol twist.

```python 
# /opt/conda/lib/python3.10/site-packages/unsloth/trainer.py
@dataclass
class UnslothTrainingArguments(TrainingArguments):
    embedding_learning_rate : Optional[float] = field(
        default = None,
        metadata = {"help" : "Different learning rates for embeddings and lm_head."}
    )
pass
```

> What are all thoses `pass`es here for? 
>
> *~ me*

So the problem that I have is not really an Unsloth problem, but a Huggingface trainer problem. 

Now don't get me wrong, I love Huggingface as every other guy. But I hate a big chunk of their documentation with passion, and also my *i-may-be-a-lil-autistic* side has a personal feud with [some of the internals](https://github.com/huggingface/transformers/pull/31798) (yeah, i know my solution is lackluster I get it, but it also makes stuff work as expected ok, I'm happy to take suggestions as how to fix it properly).

Let me sell you two Unsloth/Huggingface trainer tricks-that-should-be-common-knowledge that will make you go:
1. "Why on god's good heaven isn't this an easily available common knowledge?"
2. "I just lost my data-preprocessing-coffee-brake, why did I do that"

### Dataset processing and tokenizing
I don't know, maybe I'm just so bad at reading documentation, but in order to have any idea how to understand and modify what the trainer does with the dataset, I had to jump head first into the actual code. Googling dindu nuffin. Bear with me, pure, distilled knowledge time.

There are things the Trainer Deep State is not telling you!

1. The trainer expects a Dataset object (technically it can also be Torch dataloader but I am not going that route so have fun with your fight)
2. You have three ways now:
   1. Pass a dataset that needs to be formatted and tokenized
   2. Pass a dataset that is formatted and needs to be tokenized
   3. Pass a formatted and tokenized dataset

This is our starting point. Dataset object and a dream. Please remember to chose the split or the Trainer will throw an Exception at you.
```python
from datasets import load_dataset
dataset = load_dataset("chrisociepa/wikipedia-pl-20230401", split="train")

# Dataset({
#     features: ['id', 'url', 'title', 'text'],
#     num_rows: 1562327
# })
```

For curious souls, the whole inside-the-trainer-processing happens in: 
<details>
  <summary>_prepare_dataset</summary>

```python
# /opt/conda/lib/python3.10/site-packages/trl/trainer/sft_trainer.py
def _prepare_dataset(
        self,
        dataset,
        tokenizer,
        packing,
        dataset_text_field,
        max_seq_length,
        formatting_func,
        num_of_sequences,
        chars_per_token,
        remove_unused_columns=True,
        append_concat_token=True,
        add_special_tokens=True,
        skip_prepare_dataset=False,
):
    if dataset is None:
            raise ValueError("The dataset should not be None")
    print("Formatting func:", formatting_func)
    if skip_prepare_dataset:
        return dataset
    # If the dataset is already preprocessed (tokenized), return as-is. Only works if dataset is
    # a datasets.Dataset or datasets.IterableDataset -- not for torch Dataset
    column_names = (
        dataset.column_names if isinstance(dataset, (datasets.Dataset, datasets.IterableDataset)) else None
    )
    if column_names and "input_ids" in column_names:
        return dataset
    # check if torch dataset / dataloader and do nothing
    # see https://github.com/huggingface/trl/pull/1468 for why datasets.IterableDataset needs a separate check
    if isinstance(
        dataset, (torch.utils.data.IterableDataset, torch.utils.data.Dataset, ConstantLengthDataset)
    ) and not isinstance(dataset, datasets.IterableDataset):
        return dataset
    if not packing:
        return self._prepare_non_packed_dataloader(
            tokenizer,
            dataset,
            dataset_text_field,
            max_seq_length,
            formatting_func,
            add_special_tokens,
            remove_unused_columns,
        )
    else:
        return self._prepare_packed_dataloader(
            tokenizer,
            dataset,
            dataset_text_field,
            max_seq_length,
            num_of_sequences,
            chars_per_token,
            formatting_func,
            append_concat_token,
            add_special_tokens)
  ```
</details>

#### Pass a dataset that needs to be formatted and tokenized
Great, you chose no control over your life at all. Ok, cool. Not me though, but you do you. This is how your trainer has to be called like, look only at the commented arguments others are purely training related:

```python
trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,              # just your dataset
    dataset_text_field = "text",          # dataset feature that will be used for training
    max_seq_length = max_seq_length,    
    formatting_function = format_example, # function applied to all dataset elemens, applied with .map() method of dataset object
    dataset_num_proc = 8,                 # number of processes for tokenization and formatting

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,

        warmup_ratio = 0.1,
        num_train_epochs = 1,

        learning_rate = 5e-5,
        embedding_learning_rate = 5e-6,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.00,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
    ),
)
```

 `dataset.map()` will cache the results on the disk, so if you manage to stick through it once, you are good. Unless you want to use this data anywhere else, then you suffer through again. This is the main reason I like doing tokenization and processing myself. 

#### Pass a dataset that is formatted and needs to be tokenized

I already like it better, it does not really change much, but you have a little more control, and control is really important in programing. You can format your dataset like this:

```python
def prepare_example(example):
    # just adding EOS token after text
    # this is continual pretraining, and we want to tell the
    # model when it should stop
    return {"text": example["text"] + tokenizer.eos_token}

dataset = load_dataset("chrisociepa/wikipedia-pl-20230401")
dataset = dataset.map(prepare_example, num_proc=8)

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,              # just your dataset
    dataset_text_field = "text",          # dataset feature that will be used for training
    max_seq_length = max_seq_length
    dataset_num_proc = 8,                 # number of processes for tokenization

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,

        warmup_ratio = 0.1,
        num_train_epochs = 1,

        learning_rate = 5e-5,
        embedding_learning_rate = 5e-6,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.00,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
    ),
)
```

We got rid of the formatting function and number of workers to process it. We are more than capable of doing formatting without abstracting it thank you. 

#### Pass a formatted and tokenized dataset
So you have a preprocessed and tokenized dataset. This is buried in code: trainer basically checks if you have `input_ids` in the dataset features. So you want a dataset that looks somewhat like this:

```
Dataset({
    features: ['id', 'url', 'title', 'text', 'input_ids'],
    num_rows: 1562327
})
```

And the way to pass it:
```python
trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,                # dataset with 'input_ids' feature

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,

        warmup_ratio = 0.1,
        num_train_epochs = 1,

        learning_rate = 5e-5,
        embedding_learning_rate = 5e-6,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.00,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
    ),
)
```

Thats it. Somehow a 20 minute google and search through github issues did not tell me that. 

### Second Trick: Weights and Biases integration
This is also SO DAMN HARD TO FIND. But there is not trick, integration is by default and it will ask you to login if you are not already. 

The way to **disable** the integration is by:
```python
import os
os.environ['WANDB_DISABLED'] = 'true'
```
It will also work in (all?) most places Weights and Biases is integrated. 

To configure the Weights and Biases, you can add some run init specific stuff into the `UnslothTrainingArguments`:
```python
UnslothTrainingArguments(
    # other args and kwargs here
    report_to="wandb",  # enable logging to W&B
    run_name="bert-base-high-lr",  # name of the W&B run (optional)
    logging_steps=1,  # how often to log to W&B
    # other args and kwargs there
)
```
Or [override](https://docs.wandb.ai/guides/integrations/huggingface#customize-wandbinit) `wandb.init()` which is called under the hood:
```python
wandb.init(
    project="amazon_sentiment_analysis",
    name="bert-base-high-lr",
    tags=["baseline", "high-lr"],
    group="bert",
)
```

More info is available at https://docs.wandb.ai/guides/integrations/huggingface. Yeah, don't worry it's Huggingface not Unsloth. It's all fancy abstractions.

## Errata

I probably wrote way more words than I should've to share these, but wanted to try this format for something basic to see how I like it. Also while writing I scraped a lot of rants as I understood I got some stuff wrong (as I always do). Hopefully I stick to writing stuff, cause it was fun.