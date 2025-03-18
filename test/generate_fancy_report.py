import json
import os
import datetime
from pathlib import Path
import base64
import re


def generate_report(json_file_path, output_dir=None):
    """
    Generate a fancy HTML report from a test results JSON file.

    Args:
        json_file_path (str): Path to the JSON file
        output_dir (str, optional): Directory to save the HTML report. Defaults to same directory as JSON.

    Returns:
        str: Path to the generated HTML report
    """
    # Read the JSON file
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract filename for the report title
    file_name = os.path.basename(json_file_path)

    # Create report timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(json_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # Create output file path
    output_file = os.path.join(
        output_dir, f"report_{os.path.splitext(file_name)[0]}.html"
    )

    # Extract system info
    system_info = data[0].get("system_info", {})

    # Extract test parameters
    test_parameters = data[1].get("test_parameters", {})

    # Extract dataset info
    dataset_info = data[2].get("dataset", {})

    # Extract test results
    test_results = {}
    for i in range(3, len(data) - 1):
        for key, value in data[i].items():
            if key.startswith("Test #"):
                test_results[key] = value

    # Extract execution log
    execution_log = data[-1].get("execution_log", [])

    # Get owl SVG content
    owl_svg_path = os.path.join(os.path.dirname(__file__), "owl.svg")
    try:
        with open(owl_svg_path, "r", encoding="utf-8") as f:
            owl_svg = f.read()
    except Exception as e:
        print(f"Warning: Failed to read owl SVG file: {e}")
        # Provide a fallback simple SVG
        owl_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  <style>
    .owl-body { fill: #4CAF50; }
    .owl-eyes { fill: #ff69b4; }
    .owl-beak { fill: #ff9800; }
    .owl-wings { fill: #388E3C; }
  </style>
  <circle class="owl-body" cx="50" cy="50" r="40" />
  <circle class="owl-eyes" cx="35" cy="40" r="10" />
  <circle class="owl-eyes" cx="65" cy="40" r="10" />
  <circle fill="white" cx="35" cy="40" r="5" />
  <circle fill="white" cx="65" cy="40" r="5" />
  <circle fill="black" cx="35" cy="40" r="2" />
  <circle fill="black" cx="65" cy="40" r="2" />
  <path class="owl-beak" d="M45,55 L55,55 L50,65 Z" />
  <path class="owl-wings" d="M20,50 Q30,70 50,80 Q70,70 80,50 Q65,60 50,65 Q35,60 20,50 Z" />
  <path fill="none" stroke="#388E3C" stroke-width="2" d="M25,35 Q30,25 35,30 M65,30 Q70,25 75,35" />
</svg>"""

    # Create the HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OwlAI Test Report - {file_name}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {{
            --primary-color: #ff69b4;
            --secondary-color: #4CAF50;
            --bg-color: #f9f9f9;
            --card-bg: white;
            --text-color: #333;
            --border-radius: 8px;
            --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            --transition: all 0.3s ease;
            --header-height: 60px;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        
        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            overflow-x: hidden;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 20px 0;
            text-align: center;
            border-radius: var(--border-radius);
            margin-bottom: 30px;
            box-shadow: var(--box-shadow);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }}
        
        header::before {{
            content: "";
            position: absolute;
            top: -10px;
            left: -10px;
            right: -10px;
            bottom: -10px;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            z-index: -1;
            filter: blur(20px);
            opacity: 0.7;
        }}
        
        .header-content {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            z-index: 1;
        }}
        
        .logo {{
            width: 60px;
            height: 60px;
            animation: float 3s ease-in-out infinite;
        }}
        
        @keyframes float {{
            0% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-10px); }}
            100% {{ transform: translateY(0px); }}
        }}
        
        header h1 {{
            font-size: 2.5rem;
            margin: 0;
        }}
        
        .timestamp {{
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 5px;
        }}
        
        .card {{
            background-color: var(--card-bg);
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            padding: 20px;
            margin-bottom: 20px;
            transition: var(--transition);
            border-left: 4px solid var(--primary-color);
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }}
        
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }}
        
        .card-title {{
            font-size: 1.5rem;
            color: var(--primary-color);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .card-title i {{
            color: var(--secondary-color);
        }}
        
        .card-content {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
        }}
        
        .info-item {{
            display: flex;
            flex-direction: column;
        }}
        
        .info-label {{
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .info-value {{
            font-weight: 600;
        }}
        
        .test-result {{
            border-left: 4px solid var(--secondary-color);
            animation: fadeIn 0.5s ease-in-out;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .question {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 15px;
            color: var(--primary-color);
            padding: 10px;
            background-color: rgba(255, 105, 180, 0.1);
            border-radius: var(--border-radius);
        }}
        
        .answer {{
            background-color: rgba(76, 175, 80, 0.1);
            padding: 15px;
            border-radius: var(--border-radius);
            margin-bottom: 15px;
            white-space: pre-line;
        }}
        
        .answer-control {{
            background-color: rgba(255, 105, 180, 0.1);
            padding: 15px;
            border-radius: var(--border-radius);
            margin-bottom: 15px;
            white-space: pre-line;
        }}
        
        .answer-time {{
            font-size: 0.9rem;
            color: #666;
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 10px;
        }}
        
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        
        .tab {{
            background-color: #eee;
            padding: 10px 20px;
            border-radius: var(--border-radius);
            cursor: pointer;
            transition: var(--transition);
        }}
        
        .tab.active {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
            animation: fadeIn 0.5s ease-in-out;
        }}
        
        .progress-bar {{
            height: 6px;
            background-color: #eee;
            border-radius: 3px;
            overflow: hidden;
            margin-top: 5px;
        }}
        
        .progress {{
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            width: 0;
            transition: width 1s ease-in-out;
        }}
        
        footer {{
            text-align: center;
            margin-top: 30px;
            padding: 20px 0;
            border-top: 1px solid #eee;
            color: #666;
        }}
        
        /* Responsive styles */
        @media screen and (max-width: 768px) {{
            .card-content {{
                grid-template-columns: 1fr;
            }}
            
            header h1 {{
                font-size: 1.8rem;
            }}
            
            .header-content {{
                flex-direction: column;
                gap: 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-content">
                <div class="logo">
                    {owl_svg}
                </div>
                <div>
                    <h1>OwlAI Test Report</h1>
                    <p class="timestamp">Generated on {timestamp}</p>
                </div>
            </div>
        </header>
        
        <div class="tabs">
            <div class="tab active" data-tab="overview">Overview</div>
            <div class="tab" data-tab="system">System Info</div>
            <div class="tab" data-tab="parameters">Test Parameters</div>
            <div class="tab" data-tab="results">Test Results</div>
            <div class="tab" data-tab="execution">Execution Log</div>
        </div>
        
        <div class="tab-content active" id="overview">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fas fa-info-circle"></i>
                        Overview
                    </div>
                </div>
                <div class="card-content">
                    <div class="info-item">
                        <div class="info-label">File</div>
                        <div class="info-value">{file_name}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Model</div>
                        <div class="info-value">{test_parameters.get('model_provider', 'N/A')} / {test_parameters.get('model_name', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Questions</div>
                        <div class="info-value">{len(test_results)}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Total Runtime</div>
                        <div class="info-value">{next((item.get('Total time', 'N/A') for item in execution_log if 'Total time' in item), 'N/A')}</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="tab-content" id="system">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fas fa-desktop"></i>
                        System Information
                    </div>
                </div>
                <div class="card-content">
                    <div class="info-item">
                        <div class="info-label">OS</div>
                        <div class="info-value">{system_info.get('OS', 'N/A')} {system_info.get('OS Release', 'N/A')} - {system_info.get('OS Version', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">CPU</div>
                        <div class="info-value">{system_info.get('CPU', {}).get('Model', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">CPU Cores</div>
                        <div class="info-value">{system_info.get('CPU', {}).get('Cores', 'N/A')} cores / {system_info.get('CPU', {}).get('Threads', 'N/A')} threads</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Memory</div>
                        <div class="info-value">{system_info.get('Memory', {}).get('Total (GB)', 'N/A')} GB (Available: {system_info.get('Memory', {}).get('Available (GB)', 'N/A')} GB)</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Disk</div>
                        <div class="info-value">{system_info.get('Disk', {}).get('Total (GB)', 'N/A')} GB (Free: {system_info.get('Disk', {}).get('Free (GB)', 'N/A')} GB)</div>
                    </div>
"""

    # Add GPU information if available
    gpu_info = system_info.get("GPU", [])
    if gpu_info:
        for i, gpu in enumerate(gpu_info):
            html_content += f"""
                    <div class="info-item">
                        <div class="info-label">GPU {i+1}</div>
                        <div class="info-value">{gpu.get('Name', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">GPU Memory</div>
                        <div class="info-value">{gpu.get('Memory Total (GB)', 'N/A')} GB (Free: {gpu.get('Memory Free (GB)', 'N/A')} GB)</div>
                    </div>
"""

    html_content += """
                </div>
            </div>
        </div>
        
        <div class="tab-content" id="parameters">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fas fa-cogs"></i>
                        Test Parameters
                    </div>
                </div>
                <div class="card-content">
"""

    # Add test parameters
    for key, value in test_parameters.items():
        # Format the value for display
        if isinstance(value, list):
            display_value = ", ".join(value) if value else "None"
        elif isinstance(value, dict):
            display_value = json.dumps(value, indent=2)
        else:
            display_value = str(value) if value is not None else "None"

        html_content += f"""
                    <div class="info-item">
                        <div class="info-label">{key.replace('_', ' ').title()}</div>
                        <div class="info-value">{display_value}</div>
                    </div>
"""

    html_content += """
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fas fa-database"></i>
                        Dataset Information
                    </div>
                </div>
                <div class="card-content">
"""

    # Add dataset information
    for key, value in dataset_info.items():
        # Format the value for display
        if isinstance(value, list):
            if key == "questions":
                display_value = f"{len(value)} questions"
            else:
                display_value = ", ".join(value) if value else "None"
        elif isinstance(value, dict):
            display_value = json.dumps(value, indent=2)
        else:
            display_value = str(value) if value is not None else "None"

        html_content += f"""
                    <div class="info-item">
                        <div class="info-label">{key.replace('_', ' ').title()}</div>
                        <div class="info-value">{display_value}</div>
                    </div>
"""

    if "questions" in dataset_info and dataset_info["questions"]:
        html_content += """
                    <div class="info-item" style="grid-column: 1 / -1;">
                        <div class="info-label">Questions List</div>
                        <div class="info-value" style="white-space: pre-line;">
"""
        for i, question in enumerate(dataset_info["questions"]):
            html_content += f"{i+1}. {question}\n"

        html_content += """
                        </div>
                    </div>
"""

    html_content += """
                </div>
            </div>
        </div>
        
        <div class="tab-content" id="results">
"""

    # Add test results
    for test_key, test_data in test_results.items():
        question = test_data.get("question", "N/A")
        answer = test_data.get("answer", "N/A")
        answer_control = test_data.get("answer_control_llm", "N/A")
        answer_time = test_data.get("answer_time", "N/A")
        answer_time_control = test_data.get("answer_time_control_llm", "N/A")

        # Format the answers for better display (paragraph breaks)
        answer = answer.replace("\n\n", "<br><br>").replace("\n", "<br>")
        answer_control = answer_control.replace("\n\n", "<br><br>").replace(
            "\n", "<br>"
        )

        html_content += f"""
            <div class="card test-result">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fas fa-vial"></i>
                        {test_key}
                    </div>
                </div>
                <div class="question">{question}</div>
                <div class="tabs result-tabs">
                    <div class="tab active" data-result="model-answer">Model Answer</div>
                    <div class="tab" data-result="control-answer">Control LLM Answer</div>
                </div>
                <div class="result-content active" id="model-answer-{test_key.replace(' ', '-').replace('#', '')}">
                    <div class="answer">{answer}</div>
                    <div class="answer-time">
                        <i class="fas fa-clock"></i> Response time: {answer_time}
                    </div>
                </div>
                <div class="result-content" id="control-answer-{test_key.replace(' ', '-').replace('#', '')}">
                    <div class="answer-control">{answer_control}</div>
                    <div class="answer-time">
                        <i class="fas fa-clock"></i> Response time: {answer_time_control}
                    </div>
                </div>
            </div>
"""

    html_content += """
        </div>
        
        <div class="tab-content" id="execution">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fas fa-history"></i>
                        Execution Log
                    </div>
                </div>
                <div class="card-content" style="display: block;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <thead>
                            <tr style="background-color: #f5f5f5;">
                                <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Event</th>
                                <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Time</th>
                            </tr>
                        </thead>
                        <tbody>
"""

    # Add execution log entries
    for entry in execution_log:
        for event, time in entry.items():
            html_content += f"""
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 10px;">{event}</td>
                                <td style="padding: 10px;">{time}</td>
                            </tr>
"""

    html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <footer>
            <p>OwlAI Test Report &copy; {year} - Generated by OwlAI</p>
        </footer>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Tab switching functionality
            const tabs = document.querySelectorAll('.tabs .tab');
            tabs.forEach(tab => {
                tab.addEventListener('click', function() {
                    const tabId = this.getAttribute('data-tab');
                    
                    // Remove active class from all tabs
                    tabs.forEach(t => t.classList.remove('active'));
                    
                    // Add active class to current tab
                    this.classList.add('active');
                    
                    // Hide all tab content
                    document.querySelectorAll('.tab-content').forEach(content => {
                        content.classList.remove('active');
                    });
                    
                    // Show current tab content
                    document.getElementById(tabId).classList.add('active');
                });
            });
            
            // Result tabs functionality
            document.querySelectorAll('.result-tabs .tab').forEach(tab => {
                tab.addEventListener('click', function() {
                    const resultType = this.getAttribute('data-result');
                    const resultTabs = this.closest('.result-tabs').querySelectorAll('.tab');
                    const resultContents = this.closest('.test-result').querySelectorAll('.result-content');
                    
                    // Remove active class from all result tabs
                    resultTabs.forEach(t => t.classList.remove('active'));
                    
                    // Add active class to current result tab
                    this.classList.add('active');
                    
                    // Hide all result content
                    resultContents.forEach(content => {
                        content.classList.remove('active');
                    });
                    
                    // Show current result content
                    this.closest('.test-result').querySelector(`[id^="${resultType}"]`).classList.add('active');
                });
            });
            
            // Animate progress bars
            setTimeout(() => {
                document.querySelectorAll('.progress').forEach(progress => {
                    const width = progress.getAttribute('data-width');
                    progress.style.width = width;
                });
            }, 300);
            
            // Animation on scroll
            const animateElements = document.querySelectorAll('.card');
            
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.style.opacity = 1;
                        entry.target.style.transform = 'translateY(0)';
                    }
                });
            }, {
                threshold: 0.1
            });
            
            animateElements.forEach(element => {
                element.style.opacity = 0;
                element.style.transform = 'translateY(20px)';
                element.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                observer.observe(element);
            });
        });
    </script>
</body>
</html>
""".replace(
        "{year}", str(datetime.datetime.now().year)
    )

    # Write the HTML file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Report generated successfully: {output_file}")
    return output_file


def main():
    """
    Main function to run the script from command line.

    Usage:
        python generate_fancy_report.py <json_file_path> [output_dir]
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python generate_fancy_report.py <json_file_path> [output_dir]")
        return

    json_file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    generate_report(json_file_path, output_dir)


if __name__ == "__main__":
    main()
