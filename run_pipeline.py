#!/usr/bin/env python3
"""
PhotoFinder Pipeline Runner
Runs the complete photo processing and launches the web UI
"""

import sys
import os
import subprocess
import time
import threading
from pathlib import Path

def run_photo_processing(input_dir, output_dir):
    """Run the main photo processing script"""
    print("🔄 Starting photo processing...")
    try:
        result = subprocess.run([
            sys.executable, 'main.py', input_dir, output_dir
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✅ Photo processing completed successfully!")
            print(result.stdout)
        else:
            print("❌ Photo processing failed!")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running photo processing: {e}")
        return False
    
    return True

def start_web_ui(input_dir, output_dir):
    """Start the web UI in a separate thread"""
    def run_ui():
        try:
            subprocess.run([
                sys.executable, 'web_ui.py', input_dir, output_dir
            ], cwd=os.getcwd())
        except Exception as e:
            print(f"❌ Error starting web UI: {e}")
    
    print("🌐 Starting web UI...")
    ui_thread = threading.Thread(target=run_ui, daemon=True)
    ui_thread.start()
    
    # Give the server a moment to start
    time.sleep(2)
    
    return ui_thread

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_pipeline.py <input_directory> <output_directory>")
        print("Example: python run_pipeline.py ./working_dir ./sorted_output")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Validate input directory
    if not os.path.exists(input_dir):
        print(f"❌ Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)
    
    print(f"📸 PhotoFinder Pipeline")
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print("-" * 50)
    
    # Step 1: Run photo processing
    if not run_photo_processing(input_dir, output_dir):
        sys.exit(1)
    
    # Step 2: Start web UI
    ui_thread = start_web_ui(input_dir, output_dir)
    
    print("\n🎉 Pipeline completed successfully!")
    print(f"🌐 Open http://localhost:5000 in your browser to view results")
    print("📊 View your sorted photos with bounding box annotations")
    print("\nPress Ctrl+C to stop the web server")
    
    try:
        # Keep the main thread alive
        while ui_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

if __name__ == '__main__':
    main()
