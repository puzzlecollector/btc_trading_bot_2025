import openai 
from openai import OpenAI 
import os 
import numpy as np
import pandas as pd  
import asyncio 
import nest_asyncio 
import pickle 
import datetime

os.environ["OPENAI_API_KEY"] = "<MASKED>"
openai.api_key = "<MASKED>"
client = OpenAI()

btc_df = pd.read_csv("/content/drive/MyDrive/2025.01 비트코인 선물 봇 기본 실험 데이터/baseline_experiments/filtered_btc_data_with_labels_for_baseline_train.csv") 

with open("/content/drive/MyDrive/2025.01 비트코인 선물 봇 기본 실험 데이터/baseline_experiments/baseline_grouped_news.pkl", "rb") as f: 
  grouped_information = pickle.load(f) 

context = grouped_information[datetime.date(2024, 1, 19)] 

print(context)


import re 
import time 

def extract_decision(response_text):
    # Extract the analysis part
    analysis_match = re.search(r"분석:\s*(.+)", response_text)
    analysis = analysis_match.group(1).strip() if analysis_match else None
    
    # Extract the decision part
    decision_match = re.search(r"결정:\s*(0|1)", response_text)
    decision = int(decision_match.group(1)) if decision_match else None
    
    # Ensure both analysis and decision exist
    if analysis and decision is not None:
        return analysis, decision
    else:
        raise ValueError("Response is not in the required format.")

# Function to generate a response with retry mechanism
def generate_response_with_retry(background_information, max_retries=10):
    for attempt in range(max_retries):
        try:
            # Construct the prompt
            prompt = f"""
            주어진 background information을 바탕으로 내일 비트코인 가격이 상승(Long, 0) 또는 하락(Short, 1)할지 분석하고 결정해 주세요. 
            아래의 형식으로 답변을 작성하세요:
            
            출력 형식:
            분석: [background information에 대한 간략한 분석]
            결정: [Long(0) 또는 Short(1)]
            
            예시 출력:
            분석: 트럼프 행정부의 비트코인 전략적 비축 자산 발언과 최근 비트코인의 차트 움직임을 분석했을 때 내일 종가가 오늘 종가에 비해서 높을 확률이 더 크다고 생각합니다.
            결정: 0
            
            background information: {background_information}
            """

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """
                        주어진 정보를 분석하여 비트코인 가격이 상승할지(Long, 0) 하락할지(Short, 1) 결정합니다. 
                        분석과 결정은 반드시 지정된 형식으로 작성하세요.
                        """
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract the response content
            response_text = response.choices[0].message.content 
            
            # Try to extract the analysis and decision
            analysis, decision = extract_decision(response_text)
            return analysis, decision
        
        except ValueError as e:
            # Handle parsing errors and retry
            print(f"Attempt {attempt + 1}/{max_retries}: Response format invalid. Retrying...")
            time.sleep(1)  # Optional delay between retries

    # Raise an error if all retries fail
    raise RuntimeError("Failed to get a valid response after multiple retries.")

from tqdm.auto import tqdm

def generate_all_decisions(grouped_information):
  results = [] 

  for date_key in tqdm(sorted(grouped_information.keys()), desc="Processing Dates"):
    context = grouped_information[date_key] 
    try: 
      analysis, decision = generate_response_with_retry(context) 
      results.append((date_key, analysis, decision)) 
    except RuntimeError as e: 
      print(f"Failed to process date {date_key}: {e}") 
      results.append((date_key, "Error processing context", None)) 
  
  return results 

results = generate_all_decisions(grouped_information)


results_df = pd.DataFrame(results, columns=["date", "analysis", "decision"]) 

results_df.to_csv("/content/drive/MyDrive/baseline_grouped_news_decisions.csv", index=False, encoding="utf8-sig")

print("all decisions saved to csv") 

print(results_df) 

## will compare results afterwards  
## it covers from all the way up to 2025-01-24, but chart data only ranges from 2025-01-22, so exclude the final two when scoring accuracy. 
