import json
import argparse
import evaluate
import os
import re
from sklearn.metrics import accuracy_score
from openai import OpenAI
import numpy as np

BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "openai/gpt-5"

def load_benchmark_data(benchmark_file):
    """Load the benchmark ground truth file."""
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_model_results(results_file):
    """Load the model-generated results file in JSONL format."""
    results = {}
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Ensure the 'output' key exists and is not empty
            if data.get("output"):
                results[data['image']] = data['output']
    return results

def extract_selection_choice(selection_str):
    """Extract the option letter (A, B, C, D) from the selection string."""
    if not isinstance(selection_str, str):
        return None
    # Match formats like "**A.**", "A)", "A." at the beginning
    match = re.match(r'^\s*(?:\*\*)?([A-D])(?:\.|\))?', selection_str.strip())
    if match:
        return match.group(1)
    return None

def evaluate_summary(predictions, references):
    """Evaluate summary using ROUGE."""
    print("\n--- Evaluating Summary ---")
    if not predictions or not references:
        print("No data available for Summary evaluation.")
        return None

    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=predictions, references=references, use_aggregator=False)

    print(f"ROUGE-1 Score: {np.mean(rouge_results['rouge1']):.4f}")
    return {"rouge": rouge_results}

def evaluate_selection(predictions, references):
    """Evaluate selection using Balanced Accuracy."""
    print("\n--- Evaluating Selection ---")
    if not predictions or not references:
        print("No data available for Selection evaluation.")
        return None
        
    # Ensure labels are aligned
    valid_indices = [i for i, (p, r) in enumerate(zip(predictions, references)) if p is not None and r is not None]
    if not valid_indices:
        print("No valid pairs of options available for evaluation.")
        return None
        
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_references = [references[i] for i in valid_indices]

    accuracy = accuracy_score(valid_references, valid_predictions)
    print(f"Accuracy: {accuracy:.4f}")
    return {"accuracy": accuracy}

def parse_score(review):
    """
    Parses the final score from a given review string.
    The review is expected to contain a line starting with 'Final Score: '.
    """
    try:
        match = re.search(r"Final Score:\s*(\d+\.?\d*)", review)
        if match:
            score_str = match.group(1)
            score = float(score_str)
            return score
        else:
            print('Error parsing score: "Final Score" not found in review')
            print(review)
            return -1.0
    except Exception as e:
        print(f"An error occurred while parsing score: {e}")
        print('Error parsing score from review:', review)
        return -1.0

def evaluate_conversation_with_gpt(context_str, model_answer, reference_answer, client):
    """Evaluate a single conversation using a GPT model."""
    
    eval_prompt = (
        "Evaluate the model's answer based on the context, question, and reference answer. "
        "For each of the following criteria, provide a score from 1 to 10 followed by a brief explanation for your rating. "
        "Finally, provide a Final Score from 1 to 10 with an overall justification. "
        "Your output must strictly follow the format below, without any other introductory or concluding text:\n\n"
        "Helpfulness: [score] - [brief explanation]\n"
        "Relevance: [score] - [brief explanation]\n"
        "Accuracy: [score] - [brief explanation]\n"
        "Level of Detail: [score] - [brief explanation]\n"
        "Final Score: [score] - [overall justification]"
    )

    content = (f'[Context]\n{context_str}\n\n'
               f'[Reference Answer]\n{reference_answer}\n\n[End of Reference Answer]\n\n'
               f'[Model Answer]\n{model_answer}\n\n[End of Model Answer]\n\n'
               f'[System]\n{eval_prompt}\n\n')

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of an answer."},
                {"role": "user", "content": content}
            ],
            max_tokens=5000,
            temperature=0.2
        )
        review_text = response.choices[0].message.content.strip()
        score = parse_score(review_text)
        return {"review": review_text, "score": score}
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def evaluate_conversations(conversation_data, client):
    """Iterate and evaluate all conversations."""
    print("\n--- Evaluating Conversation (using GPT) ---")
    if not conversation_data:
        print("No data available for Conversation evaluation.")
        return None

    total_score = 0
    evaluated_count = 0
    detailed_results = []
    
    for i, data in enumerate(conversation_data):
        print(f"Evaluating conversation {i + 1}...")
        result = evaluate_conversation_with_gpt(
            data['context'], data['prediction'], data['reference'], client
        )
        
        if result and result['score'] != -1.0:
            total_score += result['score']
            evaluated_count += 1
            print(f"  -> GPT Score: {result['score']}")
            detailed_results.append(result)
        else:
            print("  -> Evaluation failed or score parsing error")
            detailed_results.append({"review": "Evaluation failed", "score": -1.0})

    average_score = total_score / evaluated_count if evaluated_count > 0 else 0
    print(f"\nConversation Average Score: {average_score:.2f} (based on {evaluated_count} valid evaluations)")
    
    return {"average_score": average_score, "details": detailed_results}


def main(args):
    # Load data
    benchmark_data = load_benchmark_data(args.benchmark_file)
    model_results = load_model_results(args.results_file)

    # Prepare lists for evaluation
    summary_predictions, summary_references = [], []
    selection_predictions, selection_references = [], []
    conversation_data_list = []

    for key, ground_truth in benchmark_data.items():
        if key in model_results:
            model_output = model_results[key]
            
            if "summary" in model_output and "summary" in ground_truth:
                summary_predictions.append(model_output.get("summary", ""))
                summary_references.append(ground_truth.get("summary", ""))

            if "selection" in model_output and "selection" in ground_truth:
                pred_choice = extract_selection_choice(model_output.get("selection", ""))
                ref_choice = extract_selection_choice(ground_truth.get("selection", "")[1])
                selection_predictions.append(pred_choice)
                selection_references.append(ref_choice)

            if "conversation" in model_output and "conversation" in ground_truth and "summary" in ground_truth:
                conv_pred = model_output.get("conversation", "")
                conv_ref = ground_truth.get("conversation", "")[1]
                summary_ref_as_context = ground_truth.get("summary", "")
                
                if conv_pred and conv_ref and summary_ref_as_context:
                    conversation_data_list.append({
                        "context": summary_ref_as_context,
                        "prediction": conv_pred,
                        "reference": conv_ref
                    })

    # Run evaluation and collect results
    eval_results = {}
    if 'summary' in args.tasks:
        eval_results['summary'] = evaluate_summary(summary_predictions, summary_references)
    
    if 'selection' in args.tasks:
        eval_results['selection'] = evaluate_selection(selection_predictions, selection_references)

    if 'conversation' in args.tasks:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("\n--- Skipping Conversation evaluation ---")
            print("Error: OPENAI_API_KEY environment variable not set.")
        else:
            client = OpenAI(
                base_url=BASE_URL,
                api_key=api_key
            )
            eval_results['conversation'] = evaluate_conversations(conversation_data_list, client)

    # Save results
    if args.save:
        with open(args.save, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=4, ensure_ascii=False)
        print(f"\nEvaluation results saved to: {args.save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CerebraGloss-Bench Automatic Evaluation Script")
    parser.add_argument("--benchmark_file", type=str, default="benchmark.json", help="Path to the benchmark data file.")
    parser.add_argument("--results_file", type=str, required=True, help="Path to the model output file (e.g., gpt5_results.jsonl).")
    parser.add_argument("--tasks", nargs='+', default=['summary', 'selection', 'conversation'], choices=['summary', 'selection', 'conversation'], help="Specify tasks to evaluate. Default is all tasks.")
    parser.add_argument("--save", type=str, default=None, help="Path to save the evaluation results as a JSON file. If not provided, results are not saved.")
    
    args = parser.parse_args()
    main(args)