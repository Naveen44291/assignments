# Prompt Engineering Assignment  
**Task:** Find the country with the most repeated letter in its *official long-form English name.*

---

## 🧠 Prompt Used

You are a precise assistant that uses official English long-form country names only.

**Question:**  
What country has the same letter repeated the most in its name?

**Examples:**  
1. In "Republic of the Philippines", the letter 'p' appears 3 times.  
2. In "Federated States of Micronesia", the letter 'e' appears 5 times.  
Between these two, "Federated States of Micronesia" has the higher repeated letter count.  

Now, apply the same logic to *all* official long-form English country names (as recognized by the United Nations) and find the one that has any single character repeated the most across its entire name.

**Instructions:**
- Use only official long-form English country names (e.g., "Republic of the Philippines", "Federated States of Micronesia").  
- Count repeated letters across the entire name (ignore spaces, punctuation, and capitalization).  
- Compare all official country names by their highest single-letter repetition count.  
- Return only the exact official English country name that has the highest such count — nothing else.

---

## ⚙️ Model Tests and Outputs

### Model: **gpt-5**
**Response:**  
> United Kingdom of Great Britain and Northern Ireland

---

### Model: **gpt-4o**
**Response:**  
> The United Kingdom of Great Britain and Northern Ireland

---

### Model: **gpt-4o-mini**
**Response:**  
> United States of America

---

### Model: **gpt-3.5-turbo**
**Response:**  
> Democratic People's Republic of Korea

---

## 🧾 Summary of Results

| Model | Response |
|--------|-----------|
| gpt-5 | United Kingdom of Great Britain and Northern Ireland |
| gpt-4o | The United Kingdom of Great Britain and Northern Ireland |
| gpt-4o-mini | United States of America |
| gpt-3.5-turbo | Democratic People's Republic of Korea |

---

## ✅ Final Answer
Across models, the **United Kingdom of Great Britain and Northern Ireland** consistently appeared as the most likely correct answer.
