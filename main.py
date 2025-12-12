cat > main.py << 'EOF'
import sys

if __name__ == "__main__":
    if "--api" in sys.argv:
        from core import run_api
        run_api()
    elif "--gui" in sys.argv:
        from gui import run_gui
        run_gui()
    else:
        from core import main
        main()
EOF 
