#!/usr/bin/env python3
"""
Benchmark API performance for the LexoRead project.

This script benchmarks the performance of the LexoRead API under various load conditions.
It measures response times, throughput, and error rates.
"""

import os
import sys
import argparse
import logging
import json
import time
import datetime
import asyncio
import aiohttp
import statistics
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("benchmark")

# Default benchmark configuration
DEFAULT_CONFIG = {
    # API endpoints to benchmark
    "endpoints": [
        {
            "name": "Text Adaptation",
            "path": "/api/text/adapt",
            "method": "POST",
            "payload": {
                "text": "The quick brown fox jumps over the lazy dog. This is a test sentence to benchmark the text adaptation API endpoint.",
                "adaptations": {
                    "font_size": 1.2,
                    "line_spacing": 1.5
                }
            }
        },
        {
            "name": "Reading Level Assessment",
            "path": "/api/reading-level/assess",
            "method": "POST",
            "payload": {
                "text": "The quick brown fox jumps over the lazy dog. This is a test sentence to benchmark the reading level assessment API endpoint.",
                "include_features": True
            }
        }
    ],
    
    # Test parameters
    "concurrency": 10,         # Number of concurrent requests
    "duration": 30,            # Duration of the test in seconds
    "ramp_up": 5,              # Ramp-up time in seconds
    "delay": 0.1,              # Delay between requests in seconds
    "timeout": 10,             # Request timeout in seconds
    
    # API configuration
    "host": "http://localhost:8000",
    "headers": {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
}

class APIBenchmark:
    """
    Class for benchmarking API performance.
    """
    
    def __init__(self, config):
        """
        Initialize the benchmark.
        
        Args:
            config (dict): Benchmark configuration
        """
        self.config = config
        self.results = {}
        self.session = None
    
    async def run_benchmark(self):
        """
        Run the benchmark.
        
        Returns:
            dict: Benchmark results
        """
        # Create aiohttp session
        async with aiohttp.ClientSession(
            headers=self.config["headers"],
            timeout=aiohttp.ClientTimeout(total=self.config["timeout"])
        ) as self.session:
            # Run benchmark for each endpoint
            for endpoint_config in self.config["endpoints"]:
                endpoint_name = endpoint_config["name"]
                logger.info(f"Benchmarking endpoint: {endpoint_name}")
                
                # Run benchmark for the endpoint
                endpoint_results = await self.benchmark_endpoint(endpoint_config)
                
                # Store results
                self.results[endpoint_name] = endpoint_results
                
                # Log results
                logger.info(f"Endpoint: {endpoint_name}")
                logger.info(f"  Requests: {endpoint_results['total_requests']}")
                logger.info(f"  Successful: {endpoint_results['successful_requests']}")
                logger.info(f"  Failed: {endpoint_results['failed_requests']}")
                logger.info(f"  Average Response Time: {endpoint_results['avg_response_time']:.2f} ms")
                logger.info(f"  90th Percentile: {endpoint_results['percentile_90']:.2f} ms")
                logger.info(f"  95th Percentile: {endpoint_results['percentile_95']:.2f} ms")
                logger.info(f"  Throughput: {endpoint_results['throughput']:.2f} req/s")
                logger.info("")
        
        return self.results
    
    async def benchmark_endpoint(self, endpoint_config):
        """
        Benchmark a single endpoint.
        
        Args:
            endpoint_config (dict): Endpoint configuration
            
        Returns:
            dict: Benchmark results for the endpoint
        """
        endpoint_path = endpoint_config["path"]
        endpoint_method = endpoint_config["method"]
        endpoint_payload = endpoint_config.get("payload", {})
        
        # Calculate the number of total requests based on concurrency and duration
        total_requests = int(self.config["concurrency"] * (self.config["duration"] / self.config["delay"]))
        
        # Create a list to store response times
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        # Start time
        start_time = time.time()
        
        # Create tasks for concurrent requests
        tasks = []
        for _ in range(self.config["concurrency"]):
            task = asyncio.create_task(
                self.request_worker(
                    endpoint_path,
                    endpoint_method,
                    endpoint_payload,
                    response_times,
                    start_time
                )
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Process results
        for result in results:
            successful_requests += result["successful"]
            failed_requests += result["failed"]
        
        # Calculate statistics
        end_time = time.time()
        duration = end_time - start_time
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            percentile_90 = np.percentile(response_times, 90)
            percentile_95 = np.percentile(response_times, 95)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = 0
            percentile_90 = 0
            percentile_95 = 0
            min_response_time = 0
            max_response_time = 0
        
        throughput = successful_requests / duration if duration > 0 else 0
        
        # Create results
        results = {
            "total_requests": successful_requests + failed_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "duration": duration,
            "avg_response_time": avg_response_time,
            "percentile_90": percentile_90,
            "percentile_95": percentile_95,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time,
            "throughput": throughput,
            "response_times": response_times
        }
        
        return results
    
    async def request_worker(self, endpoint_path, method, payload, response_times, start_time):
        """
        Worker function to send requests to the API.
        
        Args:
            endpoint_path (str): API endpoint path
            method (str): HTTP method
            payload (dict): Request payload
            response_times (list): List to store response times
            start_time (float): Benchmark start time
            
        Returns:
            dict: Results from this worker
        """
        successful = 0
        failed = 0
        
        # Continue sending requests until the benchmark duration is reached
        while time.time() - start_time < self.config["duration"]:
            # Check if we should ramp up
            elapsed = time.time() - start_time
            if elapsed < self.config["ramp_up"]:
                # Calculate the percentage of concurrency to use
                ramp_percentage = elapsed / self.config["ramp_up"]
                if np.random.random() > ramp_percentage:
                    # Skip this request during ramp up
                    await asyncio.sleep(self.config["delay"])
                    continue
            
            # Send request
            try:
                request_start = time.time()
                
                # Build URL
                url = f"{self.config['host']}{endpoint_path}"
                
                # Send request based on method
                if method == "GET":
                    async with self.session.get(url) as response:
                        await response.text()
                        
                        # Calculate response time
                        request_end = time.time()
                        response_time = (request_end - request_start) * 1000  # Convert to milliseconds
                        
                        # Check if the request was successful
                        if response.status == 200:
                            successful += 1
                            response_times.append(response_time)
                        else:
                            failed += 1
                
                elif method == "POST":
                    async with self.session.post(url, json=payload) as response:
                        await response.text()
                        
                        # Calculate response time
                        request_end = time.time()
                        response_time = (request_end - request_start) * 1000  # Convert to milliseconds
                        
                        # Check if the request was successful
                        if response.status == 200:
                            successful += 1
                            response_times.append(response_time)
                        else:
                            failed += 1
                
                else:
                    logger.error(f"Unsupported HTTP method: {method}")
                    failed += 1
            
            except Exception as e:
                logger.error(f"Error sending request: {e}")
                failed += 1
            
            # Delay between requests
            await asyncio.sleep(self.config["delay"])
        
        return {
            "successful": successful,
            "failed": failed
        }

def generate_report(results, output_dir=None):
    """
    Generate a benchmark report with graphs.
    
    Args:
        results (dict): Benchmark results
        output_dir (str, optional): Directory to save the report
    """
    # Set up output directory
    if output_dir is None:
        output_dir = Path("./benchmark_results")
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results file
    results_file = output_dir / f"benchmark_results_{timestamp}.json"
    
    # Remove response_times from results to keep the file size reasonable
    results_copy = {}
    for endpoint, endpoint_results in results.items():
        results_copy[endpoint] = endpoint_results.copy()
        if "response_times" in results_copy[endpoint]:
            del results_copy[endpoint]["response_times"]
    
    # Write results to file
    with open(results_file, "w") as f:
        json.dump(results_copy, f, indent=2)
    
    logger.info(f"Saved results to {results_file}")
    
    # Generate graphs
    generate_graphs(results, output_dir, timestamp)
    
    # Generate HTML report
    generate_html_report(results, output_dir, timestamp)
    
    logger.info(f"Generated report in {output_dir}")

def generate_graphs(results, output_dir, timestamp):
    """
    Generate graphs for the benchmark results.
    
    Args:
        results (dict): Benchmark results
        output_dir (Path): Directory to save the graphs
        timestamp (str): Timestamp for the filenames
    """
    # Create a directory for the graphs
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(exist_ok=True)
    
    # Generate response time histogram for each endpoint
    for endpoint, endpoint_results in results.items():
        response_times = endpoint_results.get("response_times", [])
        
        if not response_times:
            continue
        
        # Create a clean endpoint name for the filename
        endpoint_filename = endpoint.lower().replace(" ", "_")
        
        # Response time histogram
        plt.figure(figsize=(10, 6))
        plt.hist(response_times, bins=50, alpha=0.7)
        plt.axvline(endpoint_results["avg_response_time"], color='r', linestyle='dashed', linewidth=1, label=f'Mean: {endpoint_results["avg_response_time"]:.2f} ms')
        plt.axvline(endpoint_results["percentile_90"], color='g', linestyle='dashed', linewidth=1, label=f'90th Percentile: {endpoint_results["percentile_90"]:.2f} ms')
        plt.axvline(endpoint_results["percentile_95"], color='y', linestyle='dashed', linewidth=1, label=f'95th Percentile: {endpoint_results["percentile_95"]:.2f} ms')
        plt.xlabel("Response Time (ms)")
        plt.ylabel("Frequency")
        plt.title(f"{endpoint} - Response Time Distribution")
        plt.legend()
        plt.grid(True)
        plt.savefig(graphs_dir / f"{endpoint_filename}_response_time_{timestamp}.png")
        plt.close()
    
    # Generate comparison bar chart for all endpoints
    if len(results) > 1:
        endpoints = list(results.keys())
        avg_response_times = [results[endpoint]["avg_response_time"] for endpoint in endpoints]
        throughputs = [results[endpoint]["throughput"] for endpoint in endpoints]
        
        # Response time comparison
        plt.figure(figsize=(12, 6))
        plt.bar(endpoints, avg_response_times, alpha=0.7)
        plt.ylabel("Average Response Time (ms)")
        plt.title("Average Response Time Comparison")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(graphs_dir / f"response_time_comparison_{timestamp}.png")
        plt.close()
        
        # Throughput comparison
        plt.figure(figsize=(12, 6))
        plt.bar(endpoints, throughputs, alpha=0.7)
        plt.ylabel("Throughput (req/s)")
        plt.title("Throughput Comparison")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(graphs_dir / f"throughput_comparison_{timestamp}.png")
        plt.close()

def generate_html_report(results, output_dir, timestamp):
    """
    Generate an HTML report for the benchmark results.
    
    Args:
        results (dict): Benchmark results
        output_dir (Path): Directory to save the report
        timestamp (str): Timestamp for the filenames
    """
    # Create HTML report
    html_report = output_dir / f"benchmark_report_{timestamp}.html"
    
    # HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LexoRead API Benchmark Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .graph {{ margin: 20px 0; }}
            .summary {{ margin: 20px 0; padding: 10px; background-color: #f5f5f5; }}
        </style>
    </head>
    <body>
        <h1>LexoRead API Benchmark Report</h1>
        <p>Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <table>
                <tr>
                    <th>Endpoint</th>
                    <th>Requests</th>
                    <th>Successful</th>
                    <th>Failed</th>
                    <th>Avg Response Time (ms)</th>
                    <th>90th Percentile (ms)</th>
                    <th>95th Percentile (ms)</th>
                    <th>Throughput (req/s)</th>
                </tr>
    """
    
    # Add results for each endpoint
    for endpoint, endpoint_results in results.items():
        html_content += f"""
                <tr>
                    <td>{endpoint}</td>
                    <td>{endpoint_results["total_requests"]}</td>
                    <td>{endpoint_results["successful_requests"]}</td>
                    <td>{endpoint_results["failed_requests"]}</td>
                    <td>{endpoint_results["avg_response_time"]:.2f}</td>
                    <td>{endpoint_results["percentile_90"]:.2f}</td>
                    <td>{endpoint_results["percentile_95"]:.2f}</td>
                    <td>{endpoint_results["throughput"]:.2f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    """
    
    # Add graphs
    html_content += """
        <h2>Graphs</h2>
    """
    
    # Add response time histogram for each endpoint
    for endpoint in results.keys():
        endpoint_filename = endpoint.lower().replace(" ", "_")
        html_content += f"""
        <div class="graph">
            <h3>{endpoint} - Response Time Distribution</h3>
            <img src="graphs/{endpoint_filename}_response_time_{timestamp}.png" alt="{endpoint} Response Time Distribution" width="800">
        </div>
        """
    
    # Add comparison graphs if there are multiple endpoints
    if len(results) > 1:
        html_content += f"""
        <div class="graph">
            <h3>Average Response Time Comparison</h3>
            <img src="graphs/response_time_comparison_{timestamp}.png" alt="Response Time Comparison" width="800">
        </div>
        
        <div class="graph">
            <h3>Throughput Comparison</h3>
            <img src="graphs/throughput_comparison_{timestamp}.png" alt="Throughput Comparison" width="800">
        </div>
        """
    
    # Close HTML
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML file
    with open(html_report, "w") as f:
        f.write(html_content)

def load_config(config_file=None):
    """
    Load benchmark configuration from a file.
    
    Args:
        config_file (str, optional): Path to the configuration file
        
    Returns:
        dict: Benchmark configuration
    """
    if config_file and os.path.exists(config_file):
        with open(config_file, "r") as f:
            try:
                config = json.load(f)
                logger.info(f"Loaded configuration from {config_file}")
                return config
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing configuration file: {e}")
    
    logger.info("Using default configuration")
    return DEFAULT_CONFIG.copy()

async def run_benchmark(args):
    """
    Run the benchmark with the given arguments.
    
    Args:
        args (argparse.Namespace): Command-line arguments
        
    Returns:
        dict: Benchmark results
    """
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command-line arguments
    if args.endpoint:
        # Filter endpoints
        config["endpoints"] = [e for e in config["endpoints"] if e["path"] == args.endpoint]
        
        if not config["endpoints"]:
            logger.error(f"No endpoints found matching {args.endpoint}")
            return None
    
    if args.concurrency:
        config["concurrency"] = args.concurrency
    
    if args.duration:
        config["duration"] = args.duration
    
    if args.host:
        config["host"] = args.host
    
    # Create benchmark instance
    benchmark = APIBenchmark(config)
    
    # Run benchmark
    results = await benchmark.run_benchmark()
    
    # Generate report
    if args.output:
        generate_report(results, args.output)
    
    return results

def main():
    """Parse arguments and run the benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark LexoRead API performance")
    
    parser.add_argument("--config", type=str,
                        help="Path to configuration file")
    
    parser.add_argument("--endpoint", type=str,
                        help="API endpoint to benchmark")
    
    parser.add_argument("--concurrency", type=int,
                        help="Number of concurrent requests")
    
    parser.add_argument("--duration", type=int,
                        help="Duration of the test in seconds")
    
    parser.add_argument("--host", type=str,
                        help="API host URL")
    
    parser.add_argument("--output", type=str,
                        help="Directory to save benchmark results")
    
    args = parser.parse_args()
    
    # Run benchmark
    asyncio.run(run_benchmark(args))

if __name__ == "__main__":
    main()
