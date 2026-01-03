#!/usr/bin/env python3
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.getcwd())
    
    from api import app
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    print("\n" + "="*60)
    print("Flight Predictor API Server")
    print("="*60)
    print(f"Starting on http://0.0.0.0:{port}")
    print("Press Ctrl+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
