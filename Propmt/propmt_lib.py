from langchain_core.prompts import ChatPromptTemplate

change_data_type = ChatPromptTemplate.from_template("""
You are a data type inference engine. 
Given metadata about DataFrame columns, analyze each column and decide if its datatype 
should be converted or left as-is. 
Output must be STRICTLY valid JSON matching the schema.:{return_instructions}

Guidelines:
1. Only choose among: object, integer, float, date, boolean. Never invent other dtypes.  
2. Be careful with tricky cases:
   - Numeric-looking strings (e.g., "125", "003") → integer unless leading zeros matter (then object).  
   - Decimals or scientific notation (e.g., "12.5", "1e-5") → float.  
   - Currency/percent (e.g., "$100", "75%") → float.  
   - Dates/times (e.g., "2020-01-01", "1 Jan 2020", "12/05/22", "2022-05-12 10:30") → date.  
   - Durations (e.g., "1h 20min", "5 days", "00:15:30") → object (as it's in string).  
   - True/False, Yes/No, Y/N, 0/1 → boolean.  
   - Mixed numeric + text (e.g., "12kg", "5ft", "abc123") → object.  
   - IDs, phone numbers, zip codes, account numbers → object (even if numeric).  
   - Empty, null, or special placeholders ("NA", "null", "nan", "-") → ignore when inferring.  
   - Categories/names (e.g., "Apple", "Red", "Male") → object.  
3. Reason must clearly justify why the suggested dtype is chosen.  
4. Do not drop or modify sample values; only analyze them.  
5. Return ONLY valid JSON list of ColumnRecommendation objects. No extra text.  

Input metadata:
{Column_metadata}
""")


feature_engineering_prompt = ChatPromptTemplate.from_template("""
You are a Python expert in Pandas feature engineering.  
You are given a DataFrame called `converted_df` and column metadata: {meta_data}  

For each column, decide if feature engineering is needed and return strictly in this schema:  
{return_instructions}  

Rules:  
1. Numeric values with units (e.g., "15 kg", "5 g"):  
   - Convert to float in a **common base unit** (SI preferred).  
   - New column: `<colname>_value`.  

2. Durations (e.g., "2h 50min", "00:15:30"):  
   - Convert to **total minutes**.  
   - New column: `<colname>_minutes`.  

3. Structured text (e.g., "12 items left", "Red-Blue"):  
   - Extract meaningful **numeric/categorical features**.  

4. Already-clean columns:  
   - No code needed, keep `remake="no"`.  

Constraints:  
- `remake="yes"` only if code is generated, else `"no"`.  
- `code` must contain valid Python code (empty if `remake="no"`).  
- Code must modify `converted_df` in-place.  
- Use vectorized Pandas operations only (no row-wise loops).  
- Must handle missing/irregular values robustly.  
- Generalize logic so it works on unseen values.
- Always make sure to import the library that you are using in the code.

Examples:  
- "Weight": ["15 kg","5 g"] → new col `Weight_value` = [15, 0.005] (kg base).  
- "Duration": ["2h 50min"] → new col `Duration_minutes` = [170].  
- "Cake" (categorical) → remake="no", code="".  
""")


target_variable_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant specialized in data analysis. Your task is to identify the most appropriate target variable in a dataset and recommend its problem type.
   Problem Statement: {problem_statement}
   Dataset Schema (Column Names): {columnnames}
Instructions:
   1. Carefully analyze all columns and the problem statement to identify the most suitable target variable.
   2. If multiple columns could be targets, select the one most aligned with the problem objective.
   3. Determine the problem type: "regression", "classification", or "clustering".
      - "regression" → numeric target you want to predict.
      - "classification" → categorical target you want to predict.
      - "clustering" → no explicit target; suggest clustering only if no clear target exists.
   4. Provide a brief justification for your choice.
   5. Output the result strictly in the following schema:
      {return_instructions}
Additional Rules:
   - Always pick exactly one column as the target.
   - If the dataset does not have a clear target, set "problem_type": "clustering" and explain why.
   - Do not include any text outside the schema.
""")

feature_selection_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant specialized in feature selection for machine learning.

You are given dataset metadata in JSON format.  
Your task is to select the most useful features, drop irrelevant/leaky ones, and rank all features.

You MUST return your output in the exact JSON structure defined below:
{return_instructions}

----------------
Dataset Metadata (JSON):
{metadata_json}
----------------

Instructions:
1. Read the problem statement & target variable carefully.  
2. Select features that are informative and non-leaky.  
3. Drop features that are IDs, highly missing, constant, or leak target info.  
4. Rank all features with scores (0–1) and give a short reason for each.  
5. Ensure `selected_features` + `dropped_features` partition all features.  
6. Output **only valid JSON** matching FeatureSelectionOutput.
""")

dataset_qa_prompt = ChatPromptTemplate.from_template("""
You are a Python / pandas expert helping a data analyst answer questions about a dataset.

Dataset schema:
{schema}

The analyst's question is:
{question}

Instructions:
1. Write a concise pandas snippet that answers the question.
2. Store the final answer in a variable called `result`.
3. Wrap all code in a single ```python ... ``` block.
4. Do NOT import any libraries — `pd`, `np`, and `df` are already available.
5. Do NOT read or write any files; operate only on `df`.
6. Keep the snippet short and focused; avoid unnecessary computation.
7. If the question cannot be answered with the available columns, set
   `result = "Cannot answer: <reason>"`.

Example:
Question: "What is the average price?"
Answer:
```python
result = df['Price'].mean()
```
""")

dashboard_chart_recommendation_prompt = ChatPromptTemplate.from_template("""
You are a data visualisation expert. Given the dataset schema below, recommend the
most insightful chart types from this list:
["distribution", "correlation", "scatter", "bar", "missing_values", "boxplot"]

Dataset schema:
{schema}

Return a JSON array of chart type strings (e.g. ["distribution", "correlation"]).
Only include chart types that would be meaningful for this dataset.
Return ONLY the JSON array — no explanation, no markdown.
""")


PROMPT_REGISTRY = {
    'change_data_type': change_data_type,
    'feature_engineering' : feature_engineering_prompt,
    'target_variable' : target_variable_prompt,
    'feature_selection' : feature_selection_prompt,
    'dataset_qa' : dataset_qa_prompt,
    'dashboard_chart_recommendation' : dashboard_chart_recommendation_prompt,
}