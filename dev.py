#!/usr/bin/env python3
"""
Development script for PhotoFinder.

Similar to 'npm run test' - runs linting, tests, and starts the app.
Cleans output directories before each run to ensure fresh state.
"""

import os
import sys
import shutil
import subprocess
import time
from pathlib import Path


def clean_output_directories(output_dir: str):
    """Clean people, animals, others, and debug folders but preserve working_dir."""
    folders_to_clean = ["people", "animals", "others", "_debug_boxes"]
    
    print("🧹 Cleaning output directories...")
    for folder in folders_to_clean:
        folder_path = os.path.join(output_dir, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"   ✓ Cleaned {folder}/")
        else:
            print(f"   - {folder}/ (doesn't exist)")


def run_linting():
    """Run code linting with flake8 and black."""
    print("\n🔍 Running code linting...")
    
    # Run flake8
    try:
        result = subprocess.run([
            sys.executable, "-m", "flake8", 
            "src/", "main.py", "web_ui.py", "tests/", 
            "--max-line-length=120"
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("   ✓ Flake8: No issues found")
        else:
            print(f"   ❌ Flake8 issues found:\n{result.stdout}")
            return False
    except Exception as e:
        print(f"   ❌ Flake8 error: {e}")
        return False
    
    # Run black --check
    try:
        result = subprocess.run([
            sys.executable, "-m", "black", 
            "--check", "--line-length=120",
            "src/", "main.py", "web_ui.py", "tests/"
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("   ✓ Black: Formatting is correct")
        else:
            print(f"   ❌ Black formatting issues:\n{result.stdout}")
            print("   💡 Run 'python -m black src/ main.py web_ui.py tests/ --line-length=120' to fix")
            return False
    except Exception as e:
        print(f"   ❌ Black error: {e}")
        return False
    
    return True


def run_tests():
    """Run the test suite."""
    print("\n🧪 Running tests...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", "-v", "--tb=short"
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("   ✓ All tests passed!")
            # Extract test results
            lines = result.stdout.split('\n')
            for line in lines:
                if 'passed' in line and 'failed' not in line and 'error' not in line:
                    print(f"   📊 {line.strip()}")
        else:
            print(f"   ❌ Tests failed:\n{result.stdout}")
            if result.stderr:
                print(f"   📋 Errors:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Test error: {e}")
        return False
    
    return True


def run_photo_processing(input_dir: str, output_dir: str):
    """Run the main photo processing."""
    print(f"\n📸 Processing photos from {input_dir} to {output_dir}...")
    
    try:
        result = subprocess.run([
            sys.executable, "main.py", input_dir, output_dir
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("   ✓ Photo processing completed!")
            # Show summary
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Done!' in line or 'Found' in line:
                    print(f"   📊 {line.strip()}")
        else:
            print(f"   ❌ Photo processing failed:\n{result.stdout}")
            if result.stderr:
                print(f"   📋 Errors:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Processing error: {e}")
        return False
    
    return True


def start_web_ui(input_dir: str, output_dir: str):
    """Start the web UI in background."""
    print(f"\n🌐 Starting web UI...")
    print(f"   📁 Working: {input_dir}")
    print(f"   📁 Output: {output_dir}")
    print(f"   🔗 Open http://localhost:5000 in your browser")
    
    try:
        # Start web UI in background
        process = subprocess.Popen([
            sys.executable, "web_ui.py", input_dir, output_dir
        ], cwd=os.getcwd())
        
        # Give it a moment to start
        time.sleep(2)
        
        if process.poll() is None:  # Still running
            print("   ✓ Web UI started successfully!")
            return process
        else:
            print("   ❌ Web UI failed to start")
            return None
    except Exception as e:
        print(f"   ❌ Web UI error: {e}")
        return None


def main():
    """Main development workflow."""
    print("🚀 PhotoFinder Development Workflow")
    print("=" * 50)
    
    # Check arguments
    if len(sys.argv) < 3:
        print("Usage: python dev.py <input_directory> <output_directory>")
        print("Example: python dev.py working_dir test_output")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Validate input directory
    if not os.path.exists(input_dir):
        print(f"❌ Input directory '{input_dir}' does not exist")
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Clean directories
    clean_output_directories(output_dir)
    
    # Step 2: Run linting
    if not run_linting():
        print("\n⚠️  Linting failed. Fix issues before continuing.")
        response = input("Continue anyway? (y/N): ").lower()
        if response != 'y':
            sys.exit(1)
    
    # Step 3: Run tests
    if not run_tests():
        print("\n❌ Tests failed. Fix issues before continuing.")
        response = input("Continue anyway? (y/N): ").lower()
        if response != 'y':
            sys.exit(1)
    
    # Step 4: Process photos
    if not run_photo_processing(input_dir, output_dir):
        print("\n❌ Photo processing failed.")
        sys.exit(1)
    
    # Step 5: Start web UI
    web_process = start_web_ui(input_dir, output_dir)
    
    if web_process:
        print("\n✅ Development environment ready!")
        print("🔗 Web UI: http://localhost:5000")
        print("📁 Categories: Working, People, Animals, Others, Both")
        print("🔄 Press Ctrl+C to stop the web UI")
        
        try:
            # Wait for web process
            web_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping web UI...")
            web_process.terminate()
            web_process.wait()
            print("✅ Development workflow completed!")
    else:
        print("\n❌ Failed to start web UI")
        sys.exit(1)


if __name__ == "__main__":
    main()
