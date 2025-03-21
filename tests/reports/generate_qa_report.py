import json
import os
from datetime import datetime


def read_json_report(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_html_report(data):
    # Extract test cases and parameters
    test_cases = {k: v for k, v in data.items() if k.startswith("Test #")}
    test_parameters = data.get("test_parameters", {})
    system_info = data.get("system_info", {})

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OWLAI QA Test Report</title>
        <style>
            :root {{
                --primary-color: #2c3e50;
                --secondary-color: #34495e;
                --accent-color: #3498db;
                --background-color: #f5f6fa;
                --text-color: #2c3e50;
                --border-color: #dcdde1;
                --owlai-color: #27ae60;
                --control-color: #e67e22;
            }}
            
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: var(--text-color);
                background-color: var(--background-color);
                padding: 20px;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            
            header {{
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid var(--border-color);
            }}
            
            h1, h2 {{
                color: var(--primary-color);
                margin-bottom: 15px;
            }}
            
            h1 {{
                font-size: 2em;
            }}
            
            .info-section {{
                background: var(--background-color);
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 30px;
            }}
            
            .test-case {{
                margin-bottom: 30px;
                padding: 20px;
                border: 1px solid var(--border-color);
                border-radius: 5px;
            }}
            
            .test-case:hover {{
                box-shadow: 0 2px 15px rgba(0,0,0,0.1);
            }}
            
            .test-number {{
                color: var(--accent-color);
                font-weight: bold;
                margin-bottom: 10px;
            }}
            
            .question {{
                font-weight: bold;
                color: var(--primary-color);
                margin-bottom: 15px;
                padding: 10px;
                background: var(--background-color);
                border-radius: 5px;
            }}
            
            .answers {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            
            .answer-section {{
                padding: 15px;
                background: white;
                border-radius: 5px;
                border: 1px solid var(--border-color);
            }}
            
            .answer-header {{
                font-weight: bold;
                padding: 8px;
                margin-bottom: 10px;
                border-radius: 4px;
                color: white;
                text-align: center;
            }}
            
            .owlai-header {{
                background-color: var(--owlai-color);
            }}
            
            .control-header {{
                background-color: var(--control-color);
            }}
            
            .answer-content {{
                white-space: pre-wrap;
                padding: 10px;
                background: var(--background-color);
                border-radius: 4px;
            }}
            
            @media (max-width: 768px) {{
                .container {{
                    padding: 15px;
                }}
                
                .test-case {{
                    padding: 15px;
                }}
                
                .answers {{
                    grid-template-columns: 1fr;
                }}
            }}
            
            .timestamp {{
                color: var(--secondary-color);
                font-size: 0.9em;
                margin-bottom: 10px;
            }}
            
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
            }}
            
            .info-card {{
                background: white;
                padding: 15px;
                border-radius: 5px;
                border: 1px solid var(--border-color);
            }}
            
            .info-card h3 {{
                color: var(--accent-color);
                margin-bottom: 10px;
            }}
            
            .parameters {{
                margin-top: 20px;
            }}
            
            .parameter-item {{
                margin-bottom: 10px;
            }}
            
            .parameter-name {{
                font-weight: bold;
                color: var(--accent-color);
            }}
            
            pre {{
                background: var(--background-color);
                padding: 10px;
                border-radius: 4px;
                overflow-x: auto;
                margin-top: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>OWLAI QA Test Report</h1>
                <div class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            </header>
            
            <div class="info-section">
                <h2>System Information</h2>
                <div class="info-grid">
                    <div class="info-card">
                        <h3>Operating System</h3>
                        <p>OS: {system_info.get('OS', 'N/A')}</p>
                        <p>Version: {system_info.get('OS Version', 'N/A')}</p>
                        <p>Release: {system_info.get('OS Release', 'N/A')}</p>
                    </div>
                    <div class="info-card">
                        <h3>CPU</h3>
                        <p>Model: {system_info.get('CPU', {}).get('Model', 'N/A')}</p>
                        <p>Cores: {system_info.get('CPU', {}).get('Cores', 'N/A')}</p>
                        <p>Threads: {system_info.get('CPU', {}).get('Threads', 'N/A')}</p>
                    </div>
                    <div class="info-card">
                        <h3>Memory</h3>
                        <p>Total: {system_info.get('Memory', {}).get('Total (GB)', 'N/A')} GB</p>
                        <p>Available: {system_info.get('Memory', {}).get('Available (GB)', 'N/A')} GB</p>
                    </div>
                </div>
            </div>
            
            <div class="info-section">
                <h2>Test Parameters</h2>
                <div class="parameters">
    """

    # Add RAG config parameters
    rag_config = test_parameters.get("rag_config", {})
    html += """
                    <div class="parameter-item">
                        <div class="parameter-name">RAG Configuration:</div>
                        <pre>"""
    for key, value in rag_config.items():
        if isinstance(value, dict):
            html += f"{key}:\n"
            for sub_key, sub_value in value.items():
                html += f"  {sub_key}: {sub_value}\n"
        else:
            html += f"{key}: {value}\n"
    html += """</pre>
                    </div>
    """

    # Add control LLM config parameters
    control_config = test_parameters.get("control_llm_config", {})
    html += """
                    <div class="parameter-item">
                        <div class="parameter-name">Control LLM Configuration:</div>
                        <pre>"""
    for key, value in control_config.items():
        html += f"{key}: {value}\n"
    html += """</pre>
                    </div>
                </div>
            </div>
            
            <div class="test-cases">
    """

    # Add test cases
    for test_number, test_data in sorted(
        test_cases.items(), key=lambda x: int(x[0].split("#")[1])
    ):
        html += f"""
                <div class="test-case">
                    <div class="test-number">{test_number}</div>
                    <div class="question">{test_data['question']}</div>
                    <div class="answers">
                        <div class="answer-section">
                            <div class="answer-header owlai-header">OWLAI Response</div>
                            <div class="answer-content">{test_data['answer']}</div>
                        </div>
                        <div class="answer-section">
                            <div class="answer-header control-header">Control LLM Response</div>
                            <div class="answer-content">{test_data['answer_control_llm']}</div>
                        </div>
                    </div>
                </div>
        """

    html += """
            </div>
        </div>
    </body>
    </html>
    """

    return html


def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Input JSON file path
    input_file = os.path.join(script_dir, "20250320-151105-qa_results.json")

    # Output HTML file path
    output_file = os.path.join(script_dir, "20250320-151105-qa_results.html")

    # Read JSON data
    data = read_json_report(input_file)

    # Generate HTML report
    html_content = generate_html_report(data)

    # Write HTML report
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Report generated successfully: {output_file}")


if __name__ == "__main__":
    main()
