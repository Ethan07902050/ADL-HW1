# Intent Classification & Slot Tagging

## Task Desciription

### Intent Classification
- Input: Text
- Output: Intent

#### Example 1
- Input: i dont like my current insurance plan and and want a new one.
- Output: insurance_change

#### Example 2
- Input: when will my american express credit card expire
- Output: expiration_date

#### Example 3
- Input: how would i get to city hall via bus
- Output: directions

### Slot Tagging
- Input: Text
- Output: Tags of each word

#### Example 1
- Input: a table today for myself and 3 others
- Output: O O B-date O B-people I-people I-people O

#### Example 2
- Input: my three children and i are in the party
- Output: B-people I-people I-people I-people I-people O O O O

## Data format

### Intent Classification
- id: Unique id
- text: Input sentence
- intent: A string that denotes the intent of the input sentence

#### Example
```
{
  "text": "i need you to book me a flight from ft lauderdale to houston on southwest",
  "intent": "book_flight",
  "id": "train-0"
}
```

### Slot Tagging
- id: Unique id
- text: A list of input tokens preprocessed from the input sentence
- tags: A list of strings, each denotes the tag of its corresponding token in the input sentence 

#### Example
```
{
  "tokens": [
    "i",
    "have",
    "three",
    "people",
    "for",
    "august",
    "seventh"
  ],
  "tags": [
    "O",
    "O",
    "B-people",
    "I-people",
    "O",
    "B-date",
    "O"
  ],
  "id": "train-0"
}
```

## Instructions

### Step 1. Download Model
```
bash download.sh
```

### Step 2. Testing
For testing intent classifiation model:
```
python test_intent.py
```
The predictions are saved in `pred.intent.csv`.

For testing slot tagging model:
```
python test_slot.py
```
The predictions are saved in `pred.slot.csv`.

### Training 
For training intent classification model:
```
python train_intent.py
```

For training slot tagging model:
```
python train_slot.py
``` 