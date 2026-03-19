# Step 2: Exploring the Data

Let's examine the data that gets passed to the Transformer with your own eyes.

Try the following in Python's interactive mode (or a Jupyter Notebook).

---

## 2.1 Starting Interactive Mode

```bash
python -i tiny_llm.py
```

Adding `-i` enters interactive mode after the program runs.
Variables like `model`, `vocab`, and `id2word` remain available for use.

---

## 2.2 Checking the Vocabulary

```python
>>> vocab
{'<pad>': 0, 'the': 1, 'cat': 2, 'sat': 3, 'on': 4, 'mat': 5, '.': 6, 'dog': 7, 'log': 8, 'saw': 9}

>>> len(vocab)
10
```

A vocabulary of just 10 words. Each word is assigned a number from 0 to 9.

---

## 2.3 Trying Tokenization

```python
>>> tokenize("the cat sat on the mat", vocab)
[1, 2, 3, 4, 1, 5]
```

The sentence is converted into a sequence of numbers. `"the"` is always `1` no matter where it appears.

Let's also verify the reverse direction:

```python
>>> [id2word[i] for i in [1, 2, 3, 4, 1, 5]]
['the', 'cat', 'sat', 'on', 'the', 'mat']
```

---

## 2.4 Checking the Shape of Training Data

```python
>>> inputs, targets = make_training_data(corpus, vocab)

>>> inputs.shape
torch.Size([28, 12])

>>> targets.shape
torch.Size([28, 12])
```

28 samples, each with a length of 12 tokens.

---

## 2.5 Looking at a Single Sample in Detail

```python
>>> inputs[0]
tensor([1, 2, 3, 4, 1, 5, 6, 1, 7, 3, 4, 1])

>>> [id2word[i.item()] for i in inputs[0]]
['the', 'cat', 'sat', 'on', 'the', 'mat', '.', 'the', 'dog', 'sat', 'on', 'the']
```

This is the input to the Transformer. Next, let's look at the corresponding target:

```python
>>> targets[0]
tensor([2, 3, 4, 1, 5, 6, 1, 7, 3, 4, 1, 8])

>>> [id2word[i.item()] for i in targets[0]]
['cat', 'sat', 'on', 'the', 'mat', '.', 'the', 'dog', 'sat', 'on', 'the', 'log']
```

Let's line up the input and target side by side:

```
Input:  the  cat  sat  on  the  mat   .  the  dog  sat  on  the
Target: cat  sat  on   the mat   .   the dog  sat  on   the log
```

You can see that at each position, the "next word" is the target.

---

## 2.6 Checking the Sliding Window

The second sample is shifted by one word:

```python
>>> [id2word[i.item()] for i in inputs[1]]
['cat', 'sat', 'on', 'the', 'mat', '.', 'the', 'dog', 'sat', 'on', 'the', 'log']
```

```
Sample 0: the cat sat on the mat .  the dog sat on the
Sample 1:     cat sat on the mat .  the dog sat on the log
Sample 2:         sat on the mat .  the dog sat on the log .
```

The window slides by one word at a time, producing 28 samples in total.

---

## 2.7 Key Takeaways So Far

- **Vocabulary (vocab)**: Simply assigning numbers to 10 words
- **Tokenization**: Converting a sentence into a sequence of numbers
- **Training data**: Sliding a 12-token window to create input-target (shifted by one) pairs
- **The model's task**: Predicting the "next word" at each position

The data is very simple. In the next step,
we'll peek inside the Transformer that processes this data.

---

Next: [Step 3: Peeking Inside the Transformer](03_explore_model.md)
