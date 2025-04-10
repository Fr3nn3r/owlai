import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path


def visualize_question_results(df, question, csv_file=None, charts_dir="data/charts"):
    """Create a visualization for a single question's results dataframe"""
    # Create output directory for charts
    os.makedirs(charts_dir, exist_ok=True)

    if csv_file:
        file_name = Path(csv_file).stem
    else:
        # Generate a hash for the question if no file provided
        import hashlib

        question_hash = hashlib.md5(question.encode()).hexdigest()[:8]
        file_name = f"question_{question_hash}"

    print(f"Creating visualization for: {file_name}")

    # Skip if empty
    if df.empty:
        print(f"  Skipping empty dataframe")
        return

    # Get unique tool names
    tool_names = (
        df["tool_name"].unique() if "tool_name" in df.columns else ["Unknown Tool"]
    )

    # Create a separate plot for each tool
    for tool_name in tool_names:
        # Filter data for this tool
        tool_df = df[df["tool_name"] == tool_name]

        # Sort by judge score to make visualization more meaningful
        tool_df = tool_df.sort_values(by="judge_score", ascending=False)

        # Create figure with larger size for readability
        plt.figure(figsize=(14, 10))

        # Get indices for x-axis (document numbers)
        indices = range(len(tool_df))

        # Create separate lists for reranked and non-reranked documents
        reranked_mask = tool_df["is_reranked"] == True

        # Create grouped bar chart
        bar_width = 0.35

        # Plot judge scores
        judge_bars = plt.bar(
            indices,
            tool_df["judge_score"],
            width=bar_width,
            label="Judge Score",
            alpha=0.8,
            color="royalblue",
        )

        # Plot reranker scores (only for reranked documents)
        if "reranker_score" in tool_df.columns:
            # Handle None values and position bars next to judge score bars
            reranker_scores = tool_df["reranker_score"].fillna(0)
            reranker_bars = plt.bar(
                [i + bar_width for i in indices],
                reranker_scores,
                width=bar_width,
                label="Reranker Score",
                alpha=0.8,
                color="orange",
            )

        # Mark reranked documents with different color background
        if "is_reranked" in tool_df.columns:
            for i, is_reranked in enumerate(tool_df["is_reranked"]):
                if is_reranked:
                    plt.axvspan(i - 0.4, i + 0.4 + bar_width, alpha=0.1, color="green")

        # Add a horizontal line at y=0 for reference
        plt.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

        # Add labels, title and legend
        plt.xlabel("Document Index", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.title(
            (
                f"Scores for Question: {question[:80]}..."
                if len(question) > 80
                else f"Scores for Question: {question}"
            ),
            fontsize=14,
            fontweight="bold",
        )
        plt.suptitle(f"Tool: {tool_name}", fontsize=12)
        plt.legend(fontsize=10)

        # Add grid for better readability
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Set x-axis ticks using shorter document IDs
        doc_ids = tool_df["docID"].apply(
            lambda x: x[-8:] if isinstance(x, str) else str(x)
        )
        plt.xticks(indices, labels=doc_ids, rotation=45, ha="right", fontsize=8)

        # Add correlation information if we have both scores
        if (
            "reranker_score" in tool_df.columns
            and sum(reranked_mask) > 1
            and not tool_df[reranked_mask]["reranker_score"].isna().all()
        ):
            reranked_df = tool_df[reranked_mask]
            # Calculate correlation between judge and reranker scores
            try:
                correlation = reranked_df["judge_score"].corr(
                    reranked_df["reranker_score"]
                )
                if not np.isnan(correlation):
                    plt.figtext(
                        0.15,
                        0.02,
                        f"Correlation between Judge and Reranker scores: {correlation:.4f}",
                        fontsize=10,
                        bbox=dict(facecolor="white", alpha=0.8),
                    )
            except Exception as e:
                print(f"Could not calculate correlation: {str(e)}")

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Make room for correlation text

        # Create a filename based on the question hash and tool name
        chart_filename = f"{charts_dir}/{file_name}_{tool_name.replace('-', '_')}.png"

        # Save the figure
        plt.savefig(chart_filename, dpi=300)
        plt.close()

        print(f"  Created chart: {chart_filename}")


def visualize_scores(reports_dir="data/reports"):
    """Create visualizations for all evaluation CSV files in reports directory."""
    # Find all CSV files
    csv_files = glob.glob(f"{reports_dir}/*.csv")

    # Create output directory for charts
    charts_dir = reports_dir.replace("reports", "charts")
    if charts_dir == reports_dir:
        charts_dir = os.path.join(os.path.dirname(reports_dir), "charts")
    os.makedirs(charts_dir, exist_ok=True)

    print(f"Found {len(csv_files)} CSV files to process")

    # Process each CSV file
    for csv_file in csv_files:
        file_name = Path(csv_file).stem
        print(f"Processing {file_name}")

        # Read the CSV
        df = pd.read_csv(csv_file)

        # Skip if empty
        if df.empty:
            print(f"  Skipping empty file: {csv_file}")
            continue

        # Extract question (should be the same for all rows)
        question = (
            df["question"].iloc[0] if "question" in df.columns else "Unknown Question"
        )

        # Process this question
        visualize_question_results(df, question, csv_file, charts_dir)

    print(f"Visualization complete. Charts saved to {charts_dir}")


if __name__ == "__main__":
    visualize_scores()
