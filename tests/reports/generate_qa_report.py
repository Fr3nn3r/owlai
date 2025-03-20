import json
import os
from datetime import datetime


def read_json_report(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_html_report(data):
    # Extract test cases
    test_cases = {k: v for k, v in data.items() if k.startswith("Test #")}

    # Get system info
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
            
            h1 {{
                color: var(--primary-color);
                font-size: 2em;
                margin-bottom: 10px;
            }}
            
            .system-info {{
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
            
            .answer {{
                white-space: pre-wrap;
                padding: 15px;
                background: white;
                border-left: 4px solid var(--accent-color);
                margin-bottom: 15px;
            }}
            
            .answer-control {{
                white-space: pre-wrap;
                padding: 15px;
                background: white;
                border-left: 4px solid var(--secondary-color);
            }}
            
            @media (max-width: 768px) {{
                .container {{
                    padding: 15px;
                }}
                
                .test-case {{
                    padding: 15px;
                }}
            }}
            
            .timestamp {{
                color: var(--secondary-color);
                font-size: 0.9em;
                margin-bottom: 10px;
            }}
            
            .system-info-grid {{
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
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>OWLAI QA Test Report</h1>
                <div class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            </header>
            
            <div class="system-info">
                <h2>System Information</h2>
                <div class="system-info-grid">
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
                    <div class="answer">{test_data['answer']}</div>
                    <div class="answer-control">{test_data['answer_control_llm']}</div>
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
