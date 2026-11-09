

Prompt: How do I display val/train loss the best in a ACL latex report?

Why: realized it would be hard to fit everything in the two-column layout.

Incorporated as: matplotlib code for adapting the visualizations to the right column size:

```
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,cf§r4
    "ytick.labelsize": 7,
})

fig, ax = plt.subplots(figsize=(3.03, 2.0))   # ACL column width × height, in inches
ax.plot(train["step"], train["loss"], color="0.6", lw=1, label="Training")
ax.plot(val["step"], val["eval_loss"], color="black", lw=1.2, label="Validation")
ax.set_xlabel("Step")
ax.set_ylabel("MNRL loss")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.3, lw=0.5)
fig.tight_layout(pad=0.2)

out = os.path.basename(run) + "_loss.pdf"
fig.savefig(out, bbox_inches="tight")
print("Saved", out)
plt.show()
```

---------------------------------------

Prompt: How do I solve this error `MarianTokenizer has no attribute prepare_seq2seq_batch`

Why: ran into it when trying to run the mMARCO `translate.py` script with my own environment (using newer versions of the dependencies).

Incorporated as: fixed by calling tokenizer(...) directly with truncation and padding, and passing attention_mask to generate:

`tokenized_text = tokenizer(
    list(documents),
    max_length=args.max_seq_length,
    truncation=True,
    padding=True,
    return_tensors="pt")`

`translated = model.generate(
    input_ids=tokenized_text['input_ids'].to(device),
    attention_mask=tokenized_text['attention_mask'].to(device)`      


------------------------------------

Prompt: how do I best reconstruct the split-up triplet parts into a dataset again?

Why: wasn't sure about the best data structure to use for this.

Incorporated as: defaultdict with the following parsing logic which saves each <id,sequence> into a dictionary and only recontructs triplets for ids that occur in all three files (translated_query.tsv, translated_positive.tsv, and translated_negative.tsv). 

```
from collections import defaultdict
from datasets import Dataset

def load(path):
    parts = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            i, s = line.rstrip("\n").split("\t", 1)
            parts[int(i)].append(s)
    return {i: " ".join(s) for i, s in parts.items()}

def rebuild(prefix, n):
    cols = {c: load(f"translated/translated_{prefix}_{c}.tsv") for c in ["query", "positive", "negative"]}
    keep = [i for i in range(n) if all(i in cols[c] for c in cols)]
    return Dataset.from_dict({c: [cols[c][i] for i in keep] for c in cols})
```
--------------------------------------

Prompt: how do I know which settings to choose for fp16 and bf16?

Why: they came with the tutorial training arguments, and I wasn't sure how to treat them.

Incorporated as: checked GPU version and set these parameters according to recommendation of the AI (which was: use bf16 if your GPU supports it, which mlt2gpu-2's NVIDIA A30 do):

`fp16=False,
bf16=True,`

--------------------------------------

Prompt: if I want to do some operation on a Huggingface dataset and essentially create a new dataset from the original one, how should I go about that?

Why: realized I needed to take out positive and negative query-passage pairs from the MS MARCO dataset. 

Incorporated as: this usage of the map() function, with removal of original columns: `ds_train.map(to_triplets, batched=True, remove_columns=ds_train.column_names)`


