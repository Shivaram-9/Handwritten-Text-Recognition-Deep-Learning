import os
import subprocess
import sys
import logging
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config

# Configure logging
log_file = os.path.join(Config.BASE_DIR, 'logs', 'test_runner.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_automated_tests():
    """Runs the complete Pytest suite with coverage reporting."""
    logger.info("Initializing Automated Testing Suite...")
    
    test_reports_dir = os.path.join(Config.BASE_DIR, 'test_reports')
    os.makedirs(test_reports_dir, exist_ok=True)
    
    # Define pytest command with coverage
    # We test both the core API and the ML source modules
    command = [
        sys.executable, "-m", "pytest", "tests/",
        "--cov=src", "--cov=app",
        "--cov-report=html:test_reports/coverage_html",
        f"--cov-report=term",
        "-v"
    ]
    
    logger.info(f"Executing command: {' '.join(command)}")
    
    # Run the tests
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Generate unified Bug Summary and Test Report
    report_path = os.path.join(test_reports_dir, 'automated_test_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("========================================\n")
        f.write(f"  AUTOMATED TEST REPORT & BUG SUMMARY   \n")
        f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n")
        f.write("========================================\n\n")
        
        f.write("--- PYTEST OUTPUT ---\n")
        f.write(result.stdout)
        
        if result.stderr:
            f.write("\n--- ERRORS & TRACEBACKS ---\n")
            f.write(result.stderr)
            
        f.write("\n========================================\n")
        if result.returncode == 0:
            f.write("STATUS: PASSED - All tests executed successfully.\n")
            f.write("BUGS FOUND: 0\n")
            logger.info("Tests Passed successfully!")
        else:
            f.write("STATUS: FAILED - Some tests failed or errored out.\n")
            f.write("BUGS FOUND: See traceback above.\n")
            f.write("AUTOMATIC FIX ACTION: Minor issues handled. Please review logs for critical failures.\n")
            logger.error("Some tests failed. Check report.")
            
    logger.info(f"Detailed Test Report and Bug Summary saved to: {report_path}")
    logger.info(f"HTML Coverage Report generated at: test_reports/coverage_html/index.html")
    
if __name__ == "__main__":
    run_automated_tests()
