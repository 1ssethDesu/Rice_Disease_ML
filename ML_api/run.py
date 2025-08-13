#!/usr/bin/env python3
"""
Launcher script for Rice Disease Classification API
Run this from the ML_api directory to start the application
"""

import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

if __name__ == "__main__":
    try:
        # Import and run the main application
        from main import app, Config
        import uvicorn
        
        print(f"🚀 Starting {Config.API_TITLE} v{Config.API_VERSION}")
        print(f"📍 Server: {Config.HOST}:{Config.PORT}")
        print(f"🔧 Debug mode: {Config.DEBUG}")
        print(f"🤖 Model type: {'ONNX' if Config.USE_ONNX else 'Keras'}")
        print(f"📱 API Documentation: http://{Config.HOST}:{Config.PORT}/docs")
        print("\n" + "="*50)
        
        uvicorn.run(app, host=Config.HOST, port=Config.PORT)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running this from the ML_api directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1) 