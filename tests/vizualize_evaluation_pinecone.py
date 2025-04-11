#!/usr/bin/env python3
"""
Visualization script for RAG evaluation results.
Generates bar charts comparing judge scores, reranker scores, Pinecone scores, and token counts.
Also generates charts normalized by the mean value of each series.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import hashlib

# Configuration
EVAL_FILE = "data/evaluation/pinecone-v0.2/all_evaluations_20250411_082038.csv"
OUTPUT_DIR = "data/evaluation/pinecone-v0.2/visualization"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_evaluation_data():
    """Load evaluation data from CSV file"""
    return pd.read_csv(EVAL_FILE)


def sanitize_filename(filepath: str) -> str:
    """Sanitize the filename by removing or replacing invalid characters."""
    # Split the path into directory and filename
    directory, filename = os.path.split(filepath)

    # Replace invalid characters in the filename with an underscore
    sanitized_filename = re.sub(r'[<>:"/\\|?*]', "_", filename)

    # Reconstruct the full path with the sanitized filename
    return os.path.join(directory, sanitized_filename)


def scale_scores(scores):
    """Scale scores to a 0-1 range for comparison"""
    min_val = min(scores)
    max_val = max(scores)
    if max_val == min_val:
        return [0.5 for _ in scores]  # All same value
    return [(x - min_val) / (max_val - min_val) for x in scores]


def normalize_scores_to_mean_and_std(scores):
    """Normalize scores to have a mean of 0 and a standard deviation of 1."""
    mean_val = np.mean(scores)
    std_val = np.std(scores)
    if std_val == 0:
        return [0 for _ in scores]  # All same value
    return [(x - mean_val) / std_val for x in scores]


def create_bar_chart(data, question, output_file, normalize=False):
    """Create bar chart comparing judge, reranker, Pinecone scores, and token counts"""
    # Filter data for the specific question
    question_data = data[data["question"] == question]

    # Sort by judge score
    question_data = question_data.sort_values(by="judge_score", ascending=False)

    # Extract scores, leaving missing values as NaN for rerank_scores
    judge_scores = question_data["judge_score"].fillna(0).tolist()
    rerank_scores = question_data["reranker_score"].fillna(0).tolist()
    pc_scores = question_data["pc_score"].fillna(0).tolist()
    token_counts = question_data["token_count"].fillna(0).tolist()

    # Normalize scores if required
    if normalize:
        judge_scores = normalize_scores_to_mean_and_std(judge_scores)
        rerank_scores = normalize_scores_to_mean_and_std(rerank_scores)
        pc_scores = normalize_scores_to_mean_and_std(pc_scores)
        token_counts = normalize_scores_to_mean_and_std(token_counts)
    else:
        # Scale scores
        judge_scores = scale_scores(judge_scores)
        rerank_scores = scale_scores(rerank_scores)
        pc_scores = scale_scores(pc_scores)
        token_counts = scale_scores(token_counts)

    # Create figure
    plt.figure(figsize=(16, 6))
    title_suffix = " (Normalized by Mean)" if normalize else ""
    plt.title(f"Scores Comparison for Question: {question}{title_suffix}")

    # Bar width
    bar_width = 0.35

    # Set position of bar on X axis
    r1 = np.arange(len(judge_scores))
    r2 = [x + bar_width for x in r1]

    # Make the plot
    plt.bar(
        r1,
        judge_scores,
        color="lightblue",
        edgecolor="grey",
        width=bar_width,
        label="Judge LLM",
    )
    plt.bar(
        r1,
        rerank_scores,
        bottom=judge_scores,
        color="lightgreen",
        edgecolor="grey",
        width=bar_width,
        label="Rerank Score",
    )
    plt.bar(
        r1,
        pc_scores,
        bottom=np.array(judge_scores) + np.array(rerank_scores),
        color="lightcoral",
        edgecolor="grey",
        width=bar_width,
        label="Pinecone Score",
    )
    plt.bar(
        r2,
        token_counts,
        color="lightgrey",
        edgecolor="grey",
        width=bar_width,
        label="Token Count",
    )

    # Add labels and legend
    plt.xlabel("Document", fontweight="bold")
    plt.ylabel("Score" if normalize else "Scaled Score", fontweight="bold")
    plt.xticks(
        [r + bar_width / 2 for r in range(len(judge_scores))],
        range(1, len(judge_scores) + 1),
    )
    plt.legend()

    # Sanitize the output file name
    sanitized_output_file = sanitize_filename(output_file)

    print(f"Saving figure to '{sanitized_output_file}'")

    # Save figure
    plt.tight_layout()
    plt.savefig(sanitized_output_file, dpi=300)
    plt.close()


def main():
    """Main function to generate bar charts"""
    data = load_evaluation_data()

    if data.empty:
        print(f"No data found in {EVAL_FILE}")
        return

    questions = data["question"].unique()
    print(f"Found {len(questions)} unique questions")

    for idx, question in enumerate(questions):
        # Use a hash of the question to shorten the filename
        question_hash = hashlib.md5(question.encode()).hexdigest()[:8]

        # Create standard chart
        output_file = os.path.join(OUTPUT_DIR, f"bar_chart_{idx+1}_{question_hash}.png")
        create_bar_chart(data, question, output_file)
        print(f"Created bar chart for question: {question}")

        # Create normalized chart
        output_file_normalized = os.path.join(
            OUTPUT_DIR, f"bar_chart_normalized_{idx+1}_{question_hash}.png"
        )
        create_bar_chart(data, question, output_file_normalized, normalize=True)
        print(f"Created normalized bar chart for question: {question}")

    print("Done! All visualizations saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
