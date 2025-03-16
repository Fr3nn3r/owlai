import json
import os
import sys
import datetime
from pathlib import Path


def generate_report(json_file_path, output_dir=None):
    """
    Generate a professional HTML/JS report from QA test results JSON.

    Args:
        json_file_path (str): Path to the JSON file with test results
        output_dir (str, optional): Directory to save the report. If None, saves in the same directory as JSON.

    Returns:
        str: Path to the generated HTML report
    """
    # Load the JSON data
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

    # Extract filename for the report title
    json_filename = os.path.basename(json_file_path)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine output path
    if output_dir is None:
        output_dir = os.path.dirname(json_file_path)

    output_filename = f"report_{os.path.splitext(json_filename)[0]}.html"
    output_path = os.path.join(output_dir, output_filename)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract system info, test parameters, dataset info, and test results
    system_info = next(
        (item["system_info"] for item in data if "system_info" in item), {}
    )
    test_parameters = next(
        (item["test_parameters"] for item in data if "test_parameters" in item), {}
    )
    dataset = next((item["dataset"] for item in data if "dataset" in item), {})

    # Extract test results
    test_results = {}
    for item in data:
        for key, value in item.items():
            if key.startswith("Test #"):
                test_results[key] = value

    # Extract execution log
    execution_log = next(
        (item["execution_log"] for item in data if "execution_log" in item), []
    )

    # Calculate performance metrics
    total_answers = len(test_results)
    total_time = sum(
        float(result.get("answer_time", "0:00:00").split(":")[-1])
        for result in test_results.values()
    )
    avg_time = total_time / total_answers if total_answers > 0 else 0

    # Generate HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OwlAI Test Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary: #ff69b4;
            --primary-light: #ffb6c1;
            --primary-dark: #c71585;
            --secondary: #36454f;
            --background: #f8f8f8;
            --text: #333;
            --success: #4caf50;
            --warning: #ff9800;
            --danger: #f44336;
            --info: #2196f3;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        
        body {{
            background-color: var(--background);
            color: var(--text);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background-color: var(--primary);
            color: white;
            padding: 20px 0;
            margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }}
        
        h1, h2, h3, h4 {{
            color: var(--primary-dark);
            margin-bottom: 15px;
        }}
        
        header h1 {{
            color: white;
            font-size: 32px;
            text-align: center;
        }}
        
        .card {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 30px;
            border-top: 4px solid var(--primary);
        }}
        
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
            text-align: center;
            border-bottom: 3px solid var(--primary);
        }}
        
        .metric-card h3 {{
            font-size: 18px;
            margin-bottom: 10px;
            color: var(--secondary);
        }}
        
        .metric-card .value {{
            font-size: 28px;
            font-weight: bold;
            color: var(--primary-dark);
        }}
        
        .system-info, .test-params {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        
        .system-info-card, .test-param-card {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
        }}
        
        .system-info-card h3, .test-param-card h3 {{
            border-bottom: 2px solid var(--primary-light);
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            background-color: var(--primary-light);
            color: var(--secondary);
            font-weight: bold;
        }}
        
        tr:hover {{
            background-color: #f5f5f5;
        }}
        
        .log-entry {{
            background-color: #f5f5f5;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 10px;
        }}
        
        .tabs {{
            display: flex;
            border-bottom: 1px solid #ddd;
            margin-bottom: 20px;
        }}
        
        .tab {{
            padding: 10px 20px;
            background-color: #f1f1f1;
            border: none;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.3s;
        }}
        
        .tab.active {{
            background-color: var(--primary);
            color: white;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .question-card {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid var(--primary);
        }}
        
        .question-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }}
        
        .answer-container {{
            margin-top: 15px;
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 6px;
            white-space: pre-wrap;
        }}
        
        .answer-container p {{
            margin-bottom: 10px;
        }}
        
        .comparison-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        
        .badge {{
            background-color: var(--primary-light);
            color: var(--text);
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }}
        
        .collapsible {{
            background-color: #f1f1f1;
            color: var(--secondary);
            cursor: pointer;
            padding: 18px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 16px;
            font-weight: bold;
            transition: 0.4s;
            border-radius: 4px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1px;
        }}
        
        .active, .collapsible:hover {{
            background-color: var(--primary-light);
        }}
        
        .collapsible:after {{
            content: '\\002B';
            color: var(--secondary);
            font-weight: bold;
            float: right;
            margin-left: 5px;
        }}
        
        .active:after {{
            content: "\\2212";
        }}
        
        .content {{
            padding: 0 18px;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.2s ease-out;
            background-color: white;
            border-radius: 0 0 4px 4px;
        }}
        
        .chart-container {{
            height: 400px;
            margin-bottom: 30px;
        }}

        footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px 0;
            background-color: var(--secondary);
            color: white;
        }}
        
        .footer-content {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        @media (max-width: 768px) {{
            .metrics, .system-info, .test-params, .comparison-container {{
                grid-template-columns: 1fr;
            }}
            
            .card {{
                padding: 15px;
            }}
            
            header h1 {{
                font-size: 24px;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>OwlAI Test Report</h1>
            <p style="text-align: center; color: white;">Generated on {timestamp}</p>
        </div>
    </header>
    
    <div class="container">
        <div class="metrics">
            <div class="metric-card">
                <h3>Total Questions</h3>
                <div class="value">{total_answers}</div>
            </div>
            <div class="metric-card">
                <h3>Average Response Time</h3>
                <div class="value">{avg_time:.2f}s</div>
            </div>
            <div class="metric-card">
                <h3>Model Used</h3>
                <div class="value" style="font-size: 22px;">{test_parameters.get("model_name", "N/A")}</div>
            </div>
            <div class="metric-card">
                <h3>Control Model</h3>
                <div class="value" style="font-size: 22px;">{test_parameters.get("control_llm", "N/A")}</div>
            </div>
        </div>
        
        <div class="card">
            <h2>Performance Charts</h2>
            <div class="chart-container">
                <canvas id="responseTimeChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="compareTimesChart"></canvas>
            </div>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="openTab(event, 'questionsTab')">Questions & Answers</button>
            <button class="tab" onclick="openTab(event, 'systemTab')">System Information</button>
            <button class="tab" onclick="openTab(event, 'parametersTab')">Test Parameters</button>
            <button class="tab" onclick="openTab(event, 'executionTab')">Execution Log</button>
        </div>
        
        <div id="questionsTab" class="tab-content active">
            <h2>Test Results</h2>
            
            <!-- Questions and Answers -->
"""

    # Add question cards
    for test_key, test_data in test_results.items():
        question = test_data.get("question", "")
        answer = test_data.get("answer", "")
        answer_control = test_data.get("answer_control_llm", "")
        answer_time = test_data.get("answer_time", "")
        answer_time_control = test_data.get("answer_time_control_llm", "")

        html_content += f"""
            <div class="question-card">
                <div class="question-header">
                    <h3>{test_key}</h3>
                    <span class="badge">Response time: {answer_time}</span>
                </div>
                <p><strong>Question:</strong> {question}</p>
                
                <button class="collapsible">View Answers</button>
                <div class="content">
                    <div class="comparison-container">
                        <div>
                            <h4>Model Answer ({test_parameters.get("model_name", "Test Model")})</h4>
                            <div class="answer-container">
                                {answer.replace("\n", "<br>")}
                            </div>
                        </div>
                        <div>
                            <h4>Control Answer ({test_parameters.get("control_llm", "Control Model")})</h4>
                            <div class="answer-container">
                                {answer_control.replace("\n", "<br>")}
                            </div>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <p><strong>Model response time:</strong> {answer_time}</p>
                        <p><strong>Control response time:</strong> {answer_time_control}</p>
                    </div>
                </div>
            </div>
"""

    # System tab
    html_content += """
        </div>
        
        <div id="systemTab" class="tab-content">
            <h2>System Information</h2>
            <div class="system-info">
"""

    # Add system info cards
    if system_info:
        html_content += f"""
                <div class="system-info-card">
                    <h3>OS Information</h3>
                    <p><strong>OS:</strong> {system_info.get("OS", "N/A")}</p>
                    <p><strong>OS Version:</strong> {system_info.get("OS Version", "N/A")}</p>
                    <p><strong>OS Release:</strong> {system_info.get("OS Release", "N/A")}</p>
                </div>
"""

        # CPU information
        cpu_info = system_info.get("CPU", {})
        if cpu_info:
            html_content += f"""
                <div class="system-info-card">
                    <h3>CPU Information</h3>
                    <p><strong>Model:</strong> {cpu_info.get("Model", "N/A")}</p>
                    <p><strong>Cores:</strong> {cpu_info.get("Cores", "N/A")}</p>
                    <p><strong>Threads:</strong> {cpu_info.get("Threads", "N/A")}</p>
                    <p><strong>Max Frequency:</strong> {cpu_info.get("Max Frequency (MHz)", "N/A")} MHz</p>
                </div>
"""

        # Memory information
        memory_info = system_info.get("Memory", {})
        if memory_info:
            html_content += f"""
                <div class="system-info-card">
                    <h3>Memory Information</h3>
                    <p><strong>Total:</strong> {memory_info.get("Total (GB)", "N/A")} GB</p>
                    <p><strong>Available:</strong> {memory_info.get("Available (GB)", "N/A")} GB</p>
                </div>
"""

        # Disk information
        disk_info = system_info.get("Disk", {})
        if disk_info:
            html_content += f"""
                <div class="system-info-card">
                    <h3>Disk Information</h3>
                    <p><strong>Total:</strong> {disk_info.get("Total (GB)", "N/A")} GB</p>
                    <p><strong>Used:</strong> {disk_info.get("Used (GB)", "N/A")} GB</p>
                    <p><strong>Free:</strong> {disk_info.get("Free (GB)", "N/A")} GB</p>
                </div>
"""

        # GPU information
        gpu_info = system_info.get("GPU", [])
        if gpu_info:
            for i, gpu in enumerate(gpu_info):
                html_content += f"""
                <div class="system-info-card">
                    <h3>GPU {i+1} Information</h3>
                    <p><strong>Name:</strong> {gpu.get("Name", "N/A")}</p>
                    <p><strong>Memory Total:</strong> {gpu.get("Memory Total (GB)", "N/A")} GB</p>
                    <p><strong>Memory Free:</strong> {gpu.get("Memory Free (GB)", "N/A")} GB</p>
                    <p><strong>Memory Used:</strong> {gpu.get("Memory Used (GB)", "N/A")} GB</p>
                    <p><strong>Load:</strong> {gpu.get("Load (%)", "N/A")}%</p>
                </div>
"""

    # Parameters tab
    html_content += """
            </div>
        </div>
        
        <div id="parametersTab" class="tab-content">
            <h2>Test Parameters</h2>
            <div class="test-params">
"""

    # Add test parameters
    if test_parameters:
        for key, value in test_parameters.items():
            if isinstance(value, dict):
                html_content += f"""
                <div class="test-param-card">
                    <h3>{key.replace('_', ' ').title()}</h3>
                    <ul>
"""
                for sub_key, sub_value in value.items():
                    html_content += f"""
                        <li><strong>{sub_key.replace('_', ' ').title()}:</strong> {sub_value}</li>
"""
                html_content += """
                    </ul>
                </div>
"""
            else:
                if isinstance(value, list):
                    value_str = (
                        ", ".join(str(item) for item in value) if value else "None"
                    )
                else:
                    value_str = str(value) if value else "None"

                html_content += f"""
                <div class="test-param-card">
                    <h3>{key.replace('_', ' ').title()}</h3>
                    <p>{value_str}</p>
                </div>
"""

    # Dataset info
    if dataset:
        html_content += f"""
                <div class="test-param-card">
                    <h3>Dataset Information</h3>
                    <p><strong>Input Data Folder:</strong> {dataset.get("input_data_folder", "N/A")}</p>
                    <p><strong>Number of Questions:</strong> {len(dataset.get("questions", []))}</p>
                </div>
"""

    # Execution log tab
    html_content += """
            </div>
        </div>
        
        <div id="executionTab" class="tab-content">
            <h2>Execution Log</h2>
"""

    # Add execution log entries
    if execution_log:
        for entry in execution_log:
            for key, value in entry.items():
                html_content += f"""
            <div class="log-entry">
                <strong>{key}:</strong> {value}
            </div>
"""

    # Add charts and JavaScript
    html_content += """
        </div>
    </div>
    
    <footer>
        <div class="footer-content">
            <p>Generated by OwlAI Test Report Generator</p>
        </div>
    </footer>
    
    <script>
        // Tab functionality
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            
            // Hide all tab content
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            
            // Remove "active" class from all tabs
            tablinks = document.getElementsByClassName("tab");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            
            // Show the current tab and add "active" class
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }
        
        // Collapsible functionality
        var coll = document.getElementsByClassName("collapsible");
        var i;
        
        for (i = 0; i < coll.length; i++) {
            coll[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.maxHeight) {
                    content.style.maxHeight = null;
                } else {
                    content.style.maxHeight = content.scrollHeight + "px";
                }
            });
        }
        
        // Charts
        document.addEventListener('DOMContentLoaded', function() {
            // Response time chart
            var responseTimeCtx = document.getElementById('responseTimeChart').getContext('2d');
            var responseTimeChart = new Chart(responseTimeCtx, {
                type: 'bar',
                data: {
                    labels: ["""

    # Add response time chart data
    for test_key in test_results.keys():
        html_content += f"'{test_key}', "

    html_content += """],
                    datasets: [{
                        label: 'Response Time (seconds)',
                        data: ["""

    # Add response time values
    for test_data in test_results.values():
        response_time = test_data.get("answer_time", "0:00:00")
        seconds = float(response_time.split(":")[-1])
        html_content += f"{seconds:.2f}, "

    html_content += """],
                        backgroundColor: 'rgba(255, 105, 180, 0.7)',
                        borderColor: 'rgba(199, 21, 133, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Time (seconds)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Test Questions'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Response Time by Question',
                            font: {
                                size: 18
                            }
                        }
                    }
                }
            });
            
            // Comparison chart
            var compareTimesCtx = document.getElementById('compareTimesChart').getContext('2d');
            var compareTimesChart = new Chart(compareTimesCtx, {
                type: 'bar',
                data: {
                    labels: ["""

    # Add comparison chart labels
    for test_key in test_results.keys():
        html_content += f"'{test_key}', "

    html_content += """],
                    datasets: [
                        {
                            label: 'Test Model',
                            data: ["""

    # Add test model response times
    for test_data in test_results.values():
        response_time = test_data.get("answer_time", "0:00:00")
        seconds = float(response_time.split(":")[-1])
        html_content += f"{seconds:.2f}, "

    html_content += """],
                            backgroundColor: 'rgba(255, 105, 180, 0.7)',
                            borderColor: 'rgba(199, 21, 133, 1)',
                            borderWidth: 1
                        },
                        {
                            label: 'Control Model',
                            data: ["""

    # Add control model response times
    for test_data in test_results.values():
        response_time = test_data.get("answer_time_control_llm", "0:00:00")
        seconds = float(response_time.split(":")[-1])
        html_content += f"{seconds:.2f}, "

    html_content += """],
                            backgroundColor: 'rgba(33, 150, 243, 0.7)',
                            borderColor: 'rgba(25, 118, 210, 1)',
                            borderWidth: 1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Time (seconds)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Test Questions'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Response Time Comparison',
                            font: {
                                size: 18
                            }
                        }
                    }
                }
            });
        });
    </script>
</body>
</html>"""

    # Write HTML to file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Report generated successfully: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error writing HTML file: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <path_to_json_file> [output_directory]")
        sys.exit(1)

    json_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    report_path = generate_report(json_file, output_dir)
    if report_path:
        print(f"Report generated at: {report_path}")
        # Try to open the report in the default browser
        try:
            import webbrowser

            webbrowser.open(f"file://{os.path.abspath(report_path)}")
        except:
            pass
